import logging
import psycopg
import numpy as np
from numpy.typing import NDArray


async def insert_chunk(
    conn: psycopg.AsyncConnection,
    text: str,
    vector: NDArray[np.float32],
) -> None:
    await conn.execute(
        """
            INSERT INTO chunk_data(chunk_text, embedding)
            VALUES(%s, %s);
        """,
        (
            text,
            vector,
        ),
    )


async def retrieve_chunks(
    conn: psycopg.AsyncConnection,
    query_embedding: NDArray[np.float32],
    strat_name: str = "",
    must_include: bool = False,
    top_k: int = 5,
) -> list[tuple]:
    data = []

    if must_include:
        # """
        #     SELECT chunk_data.chunk_text, chunk_data.embedding <=> %(query_vector)s as distance
        #     FROM strat_name_lookup
        #     INNER JOIN chunk_lookup ON strat_name_lookup.strat_name_id = chunk_lookup.strat_name_id
        #     INNER JOIN chunk_data ON chunk_lookup.chunk_id = chunk_data.chunk_id
        #     WHERE strat_name LIKE %(like_pattern)s
        #     ORDER BY distance
        #     LIMIT %(top_k)s;
        # """,

        cur = await conn.execute(
            """
                SELECT chunk_text, embedding <=> %(query_embedding)s as distance
                FROM chunk_data
                WHERE chunk_text ILIKE %(like_pattern)s
                ORDER BY distance
                LIMIT %(top_k)s;
            """,
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "like_pattern": f"%{strat_name}%",
            },
        )
        data = await cur.fetchall()

    else:
        cur = await conn.execute(
            """
                SELECT chunk_text, embedding <=> %(query_embedding)s as distance
                FROM chunk_data
                ORDER BY distance
                LIMIT %(top_k)s;
            """,
            {"query_embedding": query_embedding, "top_k": top_k},
        )
        data = await cur.fetchall()
        
    return data


async def store_facts(
    conn: psycopg.AsyncConnection,
    strat_name: str,
    categories: list[str],
    facts: list[str],
    context_list: list[str],
) -> None:
    # strat_name_id = await conn.execute(
    #     """
    #         SELECT strat_name_id
    #         FROM strat_name_lookup
    #         WHERE strat_name = %s;
    #     """,
    #     (strat_name,),
    # ).fetchone()[0]

    query = """
                INSERT INTO factsheets (strat_name, {columns})
                VALUES ({parameters});
            """
    columns = ",".join(categories)
    values = [strat_name] + facts + context_list
    parameters = ",".join(["%s"] * len(values))
    await conn.execute(query.format(columns=columns, parameters=parameters), values)
