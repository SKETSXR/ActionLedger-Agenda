# src/tools/mongo_toolkit.py
from typing import List, Optional, Dict
from dotenv import dotenv_values
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase

# Simple connection cache so multiple agents can share the same handle
_CONN_CACHE: Dict[str, MongoDBDatabase] = {}


def get_mongo_tools(
    llm: BaseChatModel,
    uri: Optional[str] = None,
    database: Optional[str] = None,
) -> List[BaseTool]:
    """
    Return a reusable list of MongoDB LangChain tools bound to the given LLM and DB.

    Priority for config:
      1) explicit args (uri, database)
      2) .env keys: MONGO_CLIENT / MONGODB_URI, MONGO_DB / MONGODB_DB

    Raises:
      ValueError if URI/DB are not provided via args or env.
    """
    env = dotenv_values()
    uri = uri or env.get("MONGO_CLIENT") or env.get("MONGODB_URI")
    database = database or env.get("MONGO_DB") or env.get("MONGODB_DB")

    if not uri or not database:
        raise ValueError(
            "MongoDB URI/DB required. Pass uri/database or set "
            "MONGO_CLIENT/MONGODB_URI and MONGO_DB/MONGODB_DB in the environment."
        )

    cache_key = f"{uri}::{database}"
    db = _CONN_CACHE.get(cache_key)
    if db is None:
        db = MongoDBDatabase.from_connection_string(uri, database=database)
        _CONN_CACHE[cache_key] = db

    toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()


def close_all_mongo_connections() -> None:
    """Optional: clear cached DB wrappers (for app shutdown or test teardown)."""
    _CONN_CACHE.clear()
