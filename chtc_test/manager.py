import asyncio
import os
from httpx import AsyncClient
from arq import create_pool
from arq.connections import RedisSettings

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

REDIS_SETTINGS = RedisSettings(
    host=REDIS_HOST,     
    port=REDIS_PORT,      
    password=REDIS_PASSWORD,  
)

async def process(ctx, url):
    session: AsyncClient = ctx['session']
    response = await session.get(url)
    print(f'{url}: {response.text:.80}...')
    return len(response.text)

async def startup(ctx):
    ctx['session'] = AsyncClient()

async def shutdown(ctx):
    await ctx['session'].aclose()

class WorkerSettings:
    redis_settings = REDIS_SETTINGS
    functions = [process]
    on_startup = startup
    on_shutdown = shutdown

# Optional: main function to test enqueuing jobs
if __name__ == "__main__":
    async def main():
        redis = await create_pool(REDIS_SETTINGS)
        job = await redis.enqueue_job('process', "https://github.com")
        print(f"job {job.job_id} enqueued.")
        result = await job.result()
        print(f"Result: {result}")

    asyncio.run(main())
