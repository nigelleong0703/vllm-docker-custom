# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)


class Glm4MoeModelReasoningParser(ReasoningParser):
    """
    Reasoning parser for the Glm4MoeModel model.

    The Glm4MoeModel model uses <think>...</think> tokens to denote reasoning
    text within its output. The model provides a strict switch to disable
    reasoning output via the 'enable_thinking=False' parameter. This parser
    extracts the reasoning content enclosed by <think> and </think> tokens
    from the model's output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self.assistant_token = "<|assistant|>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.assistant_token_id = self.vocab.get(self.assistant_token)
        if (
            self.think_start_token_id is None
            or self.think_end_token_id is None
            or self.assistant_token_id is None
        ):
            raise RuntimeError(
                "Glm4MoeModel reasoning parser could not locate "
                "think start/end or assistant tokens in the tokenizer!"
            )

        # Detect template style by applying chat template and checking suffix
        chat_template_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        prompt_suffix = self._get_prompt_suffix_from_kwargs(chat_template_kwargs)

        # GLM4.7 style: template adds <think> to prompt, model continues
        self.template_added_think_start = (
            self.think_start_token in prompt_suffix
            and self.think_end_token not in prompt_suffix
        )
        # Thinking disabled: template adds </think> or <think></think>
        self.template_disabled_thinking = self.think_end_token in prompt_suffix

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        GLM's chat template has <think></think> tokens after every
        <|assistant|> token. Thus, we need to check if </think> is
        after the most recent <|assistant|> token (if present).
        """
        for token_id in input_ids[::-1]:
            if token_id == self.think_end_token_id:
                return True
            elif token_id == self.assistant_token_id:
                return False
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.think_end_token_id) + 1 :]

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content
        """
        # If thinking was disabled by template, everything is content
        if self.template_disabled_thinking:
            return DeltaMessage(content=delta_text)

        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.think_start_token_id, self.think_end_token_id]
        ):
            return None

        think_start_seen = self.think_start_token_id in previous_token_ids
        think_end_seen = self.think_end_token_id in previous_token_ids
        think_start_in_delta = self.think_start_token_id in delta_token_ids
        think_end_in_delta = self.think_end_token_id in delta_token_ids

        # GLM4.7 style: <think> was added to prompt, not in generated tokens
        # Treat as if <think> was already seen
        if self.template_added_think_start:
            think_start_seen = True

        if think_start_seen:
            if think_end_in_delta:
                # </think> in delta, extract reasoning content
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif think_end_seen:
                # </think> already seen, everything is content now
                return DeltaMessage(content=delta_text)
            else:
                # No </think> yet, reasoning continues
                return DeltaMessage(reasoning=delta_text)
        elif think_start_in_delta:
            if think_end_in_delta:
                # <think> in delta, </think> in delta, extract reasoning content
                start_index = delta_text.find(self.think_start_token)
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[
                    start_index + len(self.think_start_token) : end_index
                ]
                content = delta_text[end_index + len(self.think_end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return DeltaMessage(reasoning=delta_text)
        else:
            # No thinking tokens found, just content
            return DeltaMessage(content=delta_text)

    def _get_prompt_suffix_from_kwargs(
        self, chat_template_kwargs: dict | None
    ) -> str:
        """
        Apply the chat template to get the prompt and return the suffix
        (last part after the final assistant token).
        This helps detect if <think> or </think> was added by the template.
        """
        try:
            # Build a minimal conversation to apply the template
            messages = [{"role": "user", "content": "test"}]
            kwargs = chat_template_kwargs or {}

            prompt = self.model_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
            # Get the suffix after <|assistant|>
            if self.assistant_token in prompt:
                return prompt.split(self.assistant_token)[-1]
            return ""
        except Exception as e:
            logger.debug("Failed to get prompt suffix: %s", e)
            return ""

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # If thinking was disabled by template, everything is content
        if self.template_disabled_thinking:
            return None, model_output

        # If template added <think> (GLM4.7 style), we only need </think> in output
        if self.template_added_think_start:
            if self.think_end_token not in model_output:
                return None, model_output
            reasoning, _, content = model_output.partition(self.think_end_token)
            return reasoning, content or None

        # Original logic (GLM4.5 style): require both <think> and </think>
        if (
            self.think_start_token not in model_output
            or self.think_end_token not in model_output
        ):
            return None, model_output

        # Remove <think> if present
        model_output_parts = model_output.partition(self.think_start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.think_end_token not in model_output:
            return None, model_output

        # Extract reasoning content from the model output.
        reasoning, _, content = model_output.partition(self.think_end_token)

        final_content = content or None
        return reasoning, final_content
