import subprocess as sp
import time
import argparse

# can optimize with async subprocesses
# def get_gpu_memory(hostname: str) -> list[int]:
#     command = f"ssh wjxia@{hostname}.chtc.wisc.edu nvidia-smi --query-gpu=memory.free --format=csv"
#     output = sp.check_output(command.split(), timeout=0.5, encoding="utf-8")
#     memory_free_info = output.split("\n")[:-1][1:]
#     memory_free_values = [int(x.split()[0]) for _, x in enumerate(memory_free_info)]
#     return memory_free_values

service_ids = sp.check_output(
    ["docker", "service", "ls", "-q"], encoding="utf-8"
).strip()

if service_ids:
    # Split the service IDs into a list
    service_id_list = service_ids.split("\n")

    # Remove each service
    for service_id in service_id_list:
        sp.run(["docker", "service", "rm", service_id])
        print(f"Removed service {service_id}")


HOSTNAMES = ["cosmos0001", "cosmos0002", "cosmos0003"]
MANAGER_NODE = "cosmos0003"
WORKER_REPLICAS = 2

# DATA_DIR = "/data/sample"


# service_list = []
# for host in HOSTNAMES:
#     try:
#         gpu_list = get_gpu_memory(host)
#         for i in range(len(gpu_list)):
#             service_list.append({"host": host, "gpu_id": i, "free_mem": gpu_list[i]})
#     except sp.TimeoutExpired:
#         print(f"Could not connect to {host}")


# service_list.sort(key=lambda x: x["free_mem"], reverse=True)
service_list = [
    # {
    #     "host": "cosmos0001",
    #     "gpu_id": 0,
    # },
    # {
    #     "host": "cosmos0002",
    #     "gpu_id": 0,
    # },
    {
        "host": "cosmos0003",
        "gpu_id": 0,
    },
    {
        "host": "cosmos0003",
        "gpu_id": 1,
    },
]

if WORKER_REPLICAS > len(service_list):
    raise Exception(
        f"Only {len(service_list)} gpu nodes available but {WORKER_REPLICAS} replicas were specified."
    )

sp.run("docker build -f node.Dockerfile -t bxia68/macrostrat:node .".split())

sp.run("docker push bxia68/macrostrat:node".split())

# docker build -f pgvector.Dockerfile -t bxia68/macrostrat:pgvector .

# for i in range(WORKER_REPLICAS):
#     # host_name = service_list[i]["host"]
#     host_name = "cosmos0003"
#     gpu_id = service_list[i]["gpu_id"]


# gpu_id = 1


master_command = """
docker service create \
    --replicas 1 \
    --name master \
    --restart-condition none \
    --constraint node.hostname==cosmos0003.chtc.wisc.edu \
    --mount type=bind,source=/home/wjxia/dev/data/,destination=/data \
    --with-registry-auth \
    -e WORKER_NAMES=[{worker_list}] \
    --network bridge \
    -d \
    --network macrostrat-network \
    bxia68/macrostrat:node \
    python3.10 master.py
"""

worker_command = """
docker service create \
    --replicas 1 \
    --name worker_{worker_node}_{gpu_id} \
    --restart-condition none \
    --constraint node.hostname=={worker_node}.chtc.wisc.edu \
    --with-registry-auth \
    -e DB_HOST=pgvector \
    -e GPU_ID={gpu_id} \
    --network macrostrat-network \
    --network bridge \
    -e HOST_NAME={worker_node} \
    --dns 8.8.8.8
    -d \
    bxia68/macrostrat:node \
    python3.10 worker.py
"""


llm_command = """
docker service create \
    --replicas 1 \
    --name llm_backend_{worker_node}_{gpu_id} \
    --restart-condition none \
    --constraint node.hostname=={worker_node}.chtc.wisc.edu \
    --with-registry-auth \
    -d \
    --network macrostrat-network \
    --mount type=bind,source=/home/wjxia/dev/models,destination=/models \
    -e CUDA_VISIBLE_DEVICES={gpu_id} \
    llama.cpp \
    --server --host 0.0.0.0 -m /models/c4ai-command-r-v01-Q5_K_M.gguf --n-gpu-layers 41 -c 2000
"""
# --server --host 0.0.0.0 -m /models/starling-lm-7b-alpha.Q6_K.gguf --n-gpu-layers 35
# --server --host 0.0.0.0 -m /models/neuralhermes-2.5-mistral-7b.Q6_K.gguf --n-gpu-layers 35

db_command = f"""
docker service create \
    --replicas 1 \
    --name pgvector \
    --constraint node.hostname=={MANAGER_NODE}.chtc.wisc.edu \
    --restart-condition none \
    --network macrostrat-network \
    -d \
    -p 5432:5432
    pgvector
"""

sp.run(db_command.split())

# time.sleep(1)
# for service in service_list:
#     sp.run(
#         llm_command.format(
#             worker_node=service["host"], gpu_id=service["gpu_id"]
#         ).split()
#     )
#     sp.run(
#         worker_command.format(
#             worker_node=service["host"], gpu_id=service["gpu_id"]
#         ).split()
#     )

time.sleep(5)

worker_list = [
    f"\"worker_{service['host']}_{service['gpu_id']}\"" for service in service_list
]

sp.run(master_command.format(worker_list=",".join(worker_list)).split())

# sp.run("docker service logs pgvector".split())
# sp.run("docker service logs worker_cosmos0003".split())
sp.run("docker service logs master -f".split())
