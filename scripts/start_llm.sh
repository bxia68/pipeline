
#!/bin/bash

# docker network create worker-network
docker build -t llama.cpp -f ~/dev/llama.cpp/.devops/full-cuda.Dockerfile --build-arg CUDA_VERSION=11.6.2 --build-arg UBUNTU_VERSION=20.04 ~/dev/llama.cpp
docker run --rm --name llm_backend -it --network macrostrat-network --gpus '"device=0"' -v "$(pwd)"/models:/models llama.cpp --server -m /models/starling-lm-7b-alpha.Q6_K.gguf --n-gpu-layers 35 --host 0.0.0.0