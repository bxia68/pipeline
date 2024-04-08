import asyncio
import logging
import traceback

import grpc
import workerserver_pb2
import workerserver_pb2_grpc

import psycopg
from pgvector.psycopg import register_vector_async

import os
import time

import torch
import numpy as np
from numpy.typing import NDArray

import utils.preprocessing as preprocessing
from utils.database_utils import retrieve_chunks, insert_chunk, store_facts

from embeddings.huggingface import HuggingFaceEmbedding
from llms.llamacpp import LlamaCPPLLM
from llms.base import Message, MessageRole

EMBED_BATCH_SIZE = 16
DB_HOST = os.environ["DB_HOST"]
GPU_ID = os.environ["GPU_ID"]
HOST_NAME = os.environ["HOST_NAME"]
CONNINFO = f"dbname=vector_db host={DB_HOST} user=admin password=admin port=5432"

SYSTEM_PROMPT = """
You are a helpful and knowledgeable geologist, dedicated to reading and meticulously searching for details about specific geological stratigraphic units.
If the context provided to you does not provide the answer to the question, you will say, "I don't know".
You will not invent anything that is not drawn directly from the context.
You will be as specific as possible in its answers.
"""
QUERY_PROMPT = """
Based on the provided context and without using prior knowledge, please answer the question succinctly. Ensure that every detail in your answer is explicitly mentioned in the given context. 

Context:
\"\"\"
{context_str}
\"\"\"

Question: 
\"\"\"
{query_str}
\"\"\"
"""

class Worker(workerserver_pb2_grpc.WorkerServerServicer):
    def __init__(self):
        active_device = torch.device("cuda", int(GPU_ID))
        logging.info("Using %s device.", active_device)

        self.embedding = HuggingFaceEmbedding(
            model_name="BAAI/bge-base-en",
            device=active_device,
            context_length=512,
            instruction="Represent this sentence for searching relevant passages: ",
        )

        self.llm = LlamaCPPLLM(f"llm_backend_{HOST_NAME}_{GPU_ID}:8080", 4000)

    async def set_connection(self, conn: psycopg.AsyncConnection) -> None:
        await register_vector_async(conn)
        self.conn = conn

    async def generate_embedding(
        self, text: str, is_query: bool
    ) -> NDArray[np.float32]:
        return self.embedding.get_text_embedding(text, is_query)
    
    async def generate_batch_embedding(
        self, texts: list[str], is_query: bool
    ) -> NDArray[np.float32]:
        return self.embedding.get_batch_embedding(texts, is_query)

    async def generate_response(self, query: str, context: list[str]) -> str:
        messages = [
            Message(MessageRole.SYSTEM, SYSTEM_PROMPT),
            Message(
                MessageRole.USER,
                QUERY_PROMPT.format(query_str=query, context_str="\n\n".join(context)),
            ),
        ]

        response = await self.llm.async_chat(messages, max_tokens=200)
        
        return response.message.content
        

    async def StoreFile(
        self,
        request: workerserver_pb2.FileDataRequest,
        context: grpc.aio.ServicerContext,
    ) -> workerserver_pb2.ErrorResponse:
        try:
            text = request.document_text
            split_text = preprocessing.split_by_paragraph(text)
            split_text = [preprocessing.remove_newlines(t) for t in split_text]
            split_text = preprocessing.remove_short_sentences(split_text)
            split_text = [preprocessing.split_by_sentence(p) for p in split_text]

            chunks = []
            chunk_size = self.embedding.context_length
            for paragraph in split_text:
                token_count = self.embedding.tokenizer(paragraph, return_length=True).length
                
                current_chunk = []
                current_tokens = 0
                for sentence, token in zip(paragraph, token_count):
                    if current_tokens + token >= chunk_size:
                        chunks.append(". ".join(current_chunk) + ".")
                        current_chunk = []
                        current_tokens = 0

                    if token > chunk_size:
                        continue

                    current_chunk.append(sentence)
                    current_tokens += token + 1

                if len(current_chunk) > 0:
                    chunks.append(". ".join(current_chunk) + ".")

            # async def compute_chunks(chunk_text: str) -> None:
            #     embedding = await self.generate_embedding(chunk_text, False)
            #     await insert_chunk(self.conn, chunk_text, embedding)

            # tasks = [compute_chunks(c) for c in chunks]
            # await asyncio.gather(*tasks)
            
            db_time = 0
            for i in range(0, len(chunks), EMBED_BATCH_SIZE):
                embeddings = await self.generate_batch_embedding(chunks[i: i + EMBED_BATCH_SIZE], False)
                start = time.time()
                # TODO db insert is very slow
                tasks = [insert_chunk(self.conn, chunk_text, embedding) for chunk_text, embedding in zip(chunks[i: i + EMBED_BATCH_SIZE], embeddings)]
                await asyncio.gather(*tasks)
                end = time.time()
                db_time += end - start
                
            logging.info("db insert time: %s", db_time)
            return workerserver_pb2.ErrorResponse()

        except:
            return workerserver_pb2.ErrorResponse(error=traceback.format_exc())

    async def SetQueries(
        self,
        request: workerserver_pb2.QueryRequest,
        context: grpc.aio.ServicerContext,
    ) -> workerserver_pb2.ErrorResponse:
        if len(request.queries) != len(request.categories):
            return workerserver_pb2.ErrorResponse(
                error="Length of queries and categories must be equal"
            )

        self.queries = request.queries
        self.categories = list(request.categories) + [f"{c}_context" for c in request.categories]
        logging.info("Queries set, catgories: %s", self.categories)
        return workerserver_pb2.ErrorResponse()

    async def GenerateFacts(
        self,
        request: workerserver_pb2.FactRequest,
        context: grpc.aio.ServicerContext,
    ) -> workerserver_pb2.ErrorResponse():
        if not self.queries:
            return workerserver_pb2.ErrorResponse("Error: query list has not been set.")

        try:
            facts = []
            context_list = []
            for query in self.queries:
                query = query.format(strat_name=request.strat_name)
                query_embedding = await self.generate_embedding(query, True)

                chunks = await retrieve_chunks(
                    self.conn,
                    query_embedding,
                    strat_name=request.strat_name,
                    must_include=True,
                    top_k=8,
                )

                context = [c[0] for c in chunks]
                facts.append(
                    await self.generate_response(query, context)
                )
                context_list.append(context)

            context_list = ["\n\n###\n\n".join(c) for c in context_list]
            await store_facts(self.conn, request.strat_name, self.categories, facts, context_list)
            
            return workerserver_pb2.ErrorResponse()

        except:
            return workerserver_pb2.ErrorResponse(error=traceback.format_exc())
        
    async def Heartbeat(        
        self,
        request: workerserver_pb2.StatusRequest,
        context: grpc.aio.ServicerContext,
    ):
        return workerserver_pb2.StatusResponse(status=True)


async def serve() -> None:    
    server = grpc.aio.server()

    async with await psycopg.AsyncConnection.connect(
        conninfo=CONNINFO, autocommit=True
    ) as conn:
        worker = Worker()
        await worker.set_connection(conn)

        workerserver_pb2_grpc.add_WorkerServerServicer_to_server(worker, server)

        listen_addr = "[::]:50051"
        server.add_insecure_port(listen_addr)

        logging.info("Starting server on %s", listen_addr)
        await server.start()
        await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
