FROM gpustack/runner:cuda12.9-vllm0.12.0

# Copy the modified GLM4 reasoning parser with fixes for GLM4.5/4.7 compatibility
COPY glm4_moe_reasoning_parser.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/glm4_moe_reasoning_parser.py

