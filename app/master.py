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

CONNINFO = f"dbname=output_db host=postgres user=admin password=admin port=5432"

RELATION_MAP = {
    "stratigraphic unit has lithology of": {
        "src_type": "strat",
        "dst_type": "liths"
    },
    "lithology has color of": {
        "src_type": "liths",
        "dst_type": "lith_atts",
    }
}

@dataclass
class Progress:
    output: str = "MASTER: Finished %s."
    current: int = 0

    def increment(self):
        self.current += 1
        if self.current % 10 == 0:
            logging.info(self.output, self.current)


async def read_file(queue: asyncio.Queue, consumer_count: int) -> None:
    df = pd.read_parquet("/data/formation_sample.parquet.gzip")
    for _, row in df.iterrows():
        await queue.put(row)

        await asyncio.sleep(0)  # force context switch

    for _ in range(consumer_count):
        await queue.put(None)

    logging.info("MASTER: All paragraphs have been read.")

ID_MAPPINGS = {
    "liths": pd.read_csv("liths.csv", header=None),
    "lith_atts": pd.read_csv("lith_atts.csv", header=None),
    "strat": pd.read_csv("strat.csv", header=None)
}

def find_match(name_type, name):
    df = ID_MAPPINGS[name_type]
    match = df[df[1] == name]
    if not match.empty:
        return int(match.iloc[0, 0])
    return None

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
                if item is None:
                    break
                
                response = await stub.ProcessParagraph(
                    workerserver_pb2.ParagraphRequest(paragraph=item.paragraph)
                )

                # logging.info(response)
                source_id = None
                if not response.error:
                    for triplet in response.relationships:
                        
                        relation_info = RELATION_MAP[triplet.relationship_type] 
                        src_type = relation_info["src_type"]
                        dst_type = relation_info["dst_type"]
                        
                        strat_id = None
                        lith_att_id = None
                        lith_id = None
                        
                        if src_type == "strat":
                            strat_id = find_match("strat", triplet.head)
                        if dst_type == "liths":
                            lith_id = find_match("liths", triplet.tail)
                        if src_type == "liths":
                            lith_id = find_match("liths", triplet.head)
                        if dst_type == "lith_atts":
                            lith_att_id = find_match("lith_atts", triplet.tail)
                            
                        cur = await conn.execute(
                            """
                            INSERT INTO relationships (head, tail, type, run_id, src_type, dst_type, strat_id, lith_att_id, lith_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING relationship_id;
                            """,
                            (triplet.head[:250], triplet.tail[:250], triplet.relationship_type[:100], metadata["run_id"], src_type, dst_type, strat_id, lith_att_id, lith_id)
                        )
                        
                        relationship_id = await cur.fetchone()
                        
                        if source_id == None:
                            cur = await conn.execute(
                                """
                                INSERT INTO relationships_extracted (run_id, relationship_id)
                                VALUES (%s, %s)
                                RETURNING source_id;
                                """, 
                                (metadata["run_id"], relationship_id[0])
                            )
                            source_id = await cur.fetchone()
                            if source_id:
                                source_id = source_id[0]
                        else:
                            cur = await conn.execute(
                                """
                                INSERT INTO relationships_extracted (run_id, relationship_id, source_id)
                                VALUES (%s, %s, %s);
                                """, 
                                (metadata["run_id"], relationship_id[0], source_id)
                            )
   
                else:
                    logging.info("MASTER: Worker error.")

                await conn.execute(
                    """
                    INSERT INTO sources (search_strat_name, search_strat_id, article_id, paragraph_txt)
                    VALUES (%s, %s, %s, %s);
                    """, 
                    (item.formation_name, find_match("strat", item.formation_name), item.paper_id, item.paragraph)
                )
                
                # csv dump for testing
                relationships_query = "SELECT * FROM relationships;"
                relationships_extracted_query = "SELECT * FROM relationships_extracted;"
                sources_query = "SELECT * FROM sources;"

                cur = await conn.execute(relationships_query)
                relationships_data = await cur.fetchall()
                relationships_df = pd.DataFrame(relationships_data, columns=[desc[0] for desc in cur.description])

                cur = await conn.execute(relationships_extracted_query)
                relationships_extracted_data = await cur.fetchall()
                relationships_extracted_df = pd.DataFrame(relationships_extracted_data, columns=[desc[0] for desc in cur.description])

                cur = await conn.execute(sources_query)
                sources_data = await cur.fetchall()
                sources_df = pd.DataFrame(sources_data, columns=[desc[0] for desc in cur.description])
                
                # Write dataframes to CSV
                relationships_df.to_csv("/output/relationships.csv", index=False)
                relationships_extracted_df.to_csv("/output/relationships_extracted.csv", index=False)
                sources_df.to_csv("/output/sources.csv", index=False)

                queue.task_done()
                progress.increment()

    logging.info("MASTER: Worker %s has finished processing files.", host)


async def init_db():
    async with await psycopg.AsyncConnection.connect(
        conninfo=CONNINFO, autocommit=True
    ) as conn:
        await conn.execute(
            """
            CREATE TABLE relationships (
                head character varying(256),
                type character varying(128),
                tail character varying(256),
                src_type character varying(38),
                dst_type character varying(38),
                strat_id integer,
                lith_att_id integer,
                lith_id integer,
                relationship_id serial PRIMARY KEY NOT NULL,
                run_id integer,
                strat_name_id integer
            );
            CREATE TABLE relationships_extracted (
                run_id integer NOT NULL,
                relationship_id serial NOT NULL,
                source_id serial NOT NULL,
                PRIMARY KEY (run_id, relationship_id, source_id)
            );
            CREATE TABLE sources (
                src_id serial PRIMARY KEY NOT NULL,
                search_strat_name character varying(3181),
                search_strat_id integer,
                article_id character varying(3181),
                paragraph_txt character varying(3181)
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
                logging.info("MASTER: Connected to database successfully.")
                return
        except psycopg.OperationalError:
            logging.info("MASTER: Attempt %s: Failed to connect to database.", attempt)
            await asyncio.sleep(1)

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
    logging.info("MASTER: Starting tasks.")
    await asyncio.gather(read_task, *store_tasks)

    end = time.time()
    logging.info("MASTER: Finished processing paragraphs. [%s seconds]", end - start)


if __name__ == "__main__":
    workers = json.loads(os.environ["WORKER_NAMES"])
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(workers))
