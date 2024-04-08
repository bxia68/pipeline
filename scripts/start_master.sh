#!/bin/bash

docker build -f node.Dockerfile -t node .
docker build -f master.Dockerfile -t master .
# docker run -it --network host --name master -v "$(pwd)"/data:/data master
# docker exec node python3 test_server.py 8000
docker service create --replicas 1 --name master --restart-condition none -d -e WORKER_NAMES='["cosmos0002"]' master