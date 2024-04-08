import os
import csv
import time
import json
import logging
import pandas as pd

import asyncio
import aiofiles
from dataclasses import dataclass

import psycopg

import grpc
import workerserver_pb2
import workerserver_pb2_grpc

DATA_DIR = os.environ["DATA_DIR"]
DB_HOST = os.environ["DB_HOST"]
CONNINFO = f"dbname=ouptut_db host={DB_HOST} user=admin password=admin port=5432"


@dataclass
class Progress:
    output: str = "MASTER: Finished %s."
    current: int = 0

    def increment(self):
        self.current += 1
        if self.current % 10 == 0:
            logging.info(self.output, self.current)


async def read_file(queue: asyncio.Queue, consumer_count: int) -> None:
    df = pd.read_parquet("formation_sample.parquet.gzip")
    for _, row in df.iterrows():
        await queue.put(row)

        await asyncio.sleep(0)  # force context switch

    for _ in range(consumer_count):
        await queue.put(None)

    logging.info("MASTER: All files have been read.")


async def process_paragraph(
    host: str, queue: asyncio.Queue, progress: Progress, metadata: dict
) -> None:
    async with grpc.aio.insecure_channel(f"{host}:50051") as channel:
        stub = workerserver_pb2_grpc.WorkerServerStub(channel)

        async with await psycopg.AsyncConnection.connect(
            conninfo=CONNINFO, autocommit=True
        ) as conn:

            while True:
                item = await queue.get()
                if item == None:
                    break

                response = await stub.ProcessParagraph(
                    workerserver_pb2.ParagraphRequest(document_text=item)
                )

                if not response.error:
                    for triplet in response.relationships:
                        await conn.execute(
                            f"""
                            INSERT INTO relationships (head, type, model_used, tail, run_id)
                            VALUES ({triplet.head}, {triplet.relationship_type}, {response.model_used}, {triplet.tail}, {metadata["run_id"]});
                            """
                        )
                else:
                    logging.info("MASTER: Worker error.")

                queue.task_done()
                progress.increment()

    logging.info("MASTER: Worker %s has finished processing files.", host)


async def init_db():
    async with await psycopg.AsyncConnection.connect(
        conninfo=CONNINFO, autocommit=True
    ) as conn:
        await conn.execute(
            """
                CREATE TABLE IF NOT EXISTS factsheets (
                    strat_name varchar(100),
                    PRIMARY KEY(strat_name)
                );
            """,
        )


async def connect_worker(server_address: str, max_attempts: int = 5) -> None:
    attempt = 1
    while True:
        try:
            async with grpc.aio.insecure_channel(f"{server_address}:50051") as channel:
                stub = workerserver_pb2_grpc.WorkerServerStub(channel)
                resp = await stub.Heartbeat(workerserver_pb2.StatusRequest())
                if resp.status:
                    logging.info(
                        "MASTER: Connected to %s successfully.", server_address
                    )
                    return
                else:
                    logging.info(
                        "MASTER: Worker server at %s is not ready yet.", server_address
                    )
        except grpc.RpcError:
            logging.info(
                "MASTER: Attempt %s: Failed to connect to %s.", attempt, server_address
            )
            await asyncio.sleep(2)

        attempt += 1


async def connect_db() -> None:
    attempt = 1
    while True:
        try:
            async with await psycopg.AsyncConnection.connect(conninfo=CONNINFO):
                return
        except psycopg.OperationalError:
            logging.info("MASTER: Attempt %s: Failed to connect to database.")
            await asyncio.sleep(2)

        attempt += 1


async def main(worker_list: list[str]) -> None:
    await asyncio.gather(
        *[connect_worker(worker) for worker in worker_list], connect_db()
    )

    await init_db()

    start = time.time()

    queue = asyncio.Queue()

    progress = Progress("MASTER: Processed %s paragraphs.")
    read_task = read_file(queue, len(worker_list))
    store_tasks = [
        process_paragraph(port, queue, progress, {"run_id": 1}) for port in worker_list
    ]

    await asyncio.gather(read_task, *store_tasks)

    end = time.time()
    logging.info("MASTER: Finished processing paragraphs. [%s seconds]", end - start)


if __name__ == "__main__":
    workers = json.loads(os.environ["WORKER_NAMES"])
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(workers))
