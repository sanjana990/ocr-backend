import os
import aiohttp
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials are not set in the environment")

async def get_tables_and_databases():
    """
    Fetch the list of tables and databases from Supabase.
    """
    url = f"{SUPABASE_URL}/rest/v1/rpc/get_tables_and_databases"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

    # SQL query to fetch tables and schemas
    sql_query = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY table_schema, table_name;
    """

    payload = {"sql": sql_query}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                print("✅ Retrieved tables and databases:")
                return data
            else:
                error_text = await response.text()
                print(f"❌ Failed to fetch tables and databases: {response.status} - {error_text}")
                return None

# Run the async function
if __name__ == "__main__":
    asyncio.run(get_tables_and_databases())