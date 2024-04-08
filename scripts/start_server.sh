#!/bin/bash

docker build -f node.Dockerfile -t node .
docker build -f worker.Dockerfile -t worker .
# docker run -it -p 5432:5432 --name worker worker
docker run -it --network host --name worker worker
# docker exec node python3 test_server.py 8000