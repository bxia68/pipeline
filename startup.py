import subprocess as sp
import time
import yaml

def clear_services():
    service_ids = sp.check_output(["docker", "service", "ls", "-q"], encoding="utf-8").strip()

    if service_ids:
        # Split the service IDs into a list
        service_id_list = service_ids.split("\n")

        # Remove each service
        for service_id in service_id_list:
            sp.run(["docker", "service", "rm", service_id])

    # wait for llms to free memory
    time.sleep(8)


if __name__ == "__main__":
    clear_services()
    
    # Read manager node and replicas from service.yml
    with open('service.yml', 'r') as f:
        service_config = yaml.safe_load(f)

    master_node = service_config["master_node"]
    worker_replicas = service_config["worker_replicas"]

    service_list = service_config["service_list"]

    if worker_replicas > len(service_list):
        raise Exception(
            f"Only {len(service_list)} gpu nodes available but {worker_replicas} replicas were specified."
        )

    # build and push images
    sp.run("docker build -f node.Dockerfile -t bxia68/macrostrat:node .".split())
    sp.run("docker push bxia68/macrostrat:node".split())
    sp.run("docker build -f postgres.Dockerfile -t pipeline_postgres .".split())

    master_command = """
    docker service create \
        --replicas 1 \
        --name master \
        --restart-condition none \
        --constraint node.hostname=={master_node}.chtc.wisc.edu \
        --mount type=bind,source=/home/wjxia/dev/data/,destination=/data \
        --mount type=bind,source=/home/wjxia/dev/pipeline/output,destination=/output \
        --with-registry-auth \
        -e WORKER_NAMES=[{worker_list}] \
        -d \
        --network pipeline-network \
        bxia68/macrostrat:node \
        python3.10 master.py
    """

    # worker_container_command = """
    # docker service create \
    #     --replicas 1 \
    #     --name worker_{worker_node}_{gpu_id} \
    #     --restart-condition none \
    #     --constraint node.hostname=={worker_node}.chtc.wisc.edu \
    #     --with-registry-auth \
    #     -e GPU_ID={gpu_id} \
    #     -e CUDA_VISIBLE_DEVICES={gpu_id} \
    #     --network pipeline-network \
    #     -e HOST_NAME={worker_node} \
    #     -d \
    #     bxia68/macrostrat:worker
    # """

    # llm_command = """
    # docker service create \
    #     --replicas 1 \
    #     --name llm_backend_{worker_node}_{gpu_id} \
    #     --restart-condition none \
    #     --constraint node.hostname=={worker_node}.chtc.wisc.edu \
    #     --with-registry-auth \
    #     -d \
    #     --network pipeline-network \
    #     --mount type=bind,source=/home/wjxia/dev/models,destination=/models \
    #     -e CUDA_VISIBLE_DEVICES={gpu_id} \
    #     llama.cpp \
    #     --server --host 0.0.0.0 -m /models/c4ai-command-r-v01-Q5_K_M.gguf --n-gpu-layers 41 -c 2000 -n 4000
    # """

    db_command = f"""
    docker service create \
        --replicas 1 \
        --name postgres \
        --constraint node.hostname=={master_node}.chtc.wisc.edu \
        --restart-condition none \
        --network pipeline-network \
        -d \
        -p 5432:5432 \
        pipeline_postgres
    """

    # run db
    sp.run(db_command.split())
    
    # Read services from containers.yml
    with open('containers.yml', 'r') as f:
        container_config = yaml.safe_load(f)
    containers = container_config.get("containers", {})

    # Run worker_command for each service
    for container_name, container_data in containers.items():
        for service in service_list:
            worker_command = """
            docker service create \
                --replicas 1 \
                --name {container_name}_{worker_node}_{gpu_id} \
                --restart-condition none \
                --constraint node.hostname=={worker_node}.chtc.wisc.edu \
                --with-registry-auth \
                -e GPU_ID={gpu_id} \
                -e CUDA_VISIBLE_DEVICES={gpu_id} \
                -e HOST_NAME={worker_node} \
                --network pipeline-network \
                {mount} \
                -d \
                {image} \
                {settings}
            """
            mount = ""
            if "mount" in container_data:
                mount_data = container_data["mount"].split(":")
                mount = f"--mount type=bind,source={mount_data[0]},destination={mount_data[1]}"

            sp.run(
                worker_command.format(
                    container_name=container_name,
                    gpu_id=service.get("gpu_id", ""),
                    worker_node=service.get("host", ""),
                    image=container_data.get("image", ""),
                    mount=mount,
                    settings=container_data.get("settings", "")
                ).split()
            )

    worker_list = [
        f"\"worker_{service['host']}_{service['gpu_id']}\"" for service in service_list
    ]

    # run master
    sp.run(master_command.format(master_node=master_node, worker_list=",".join(worker_list)).split())

    sp.run("docker service logs master -f".split())
