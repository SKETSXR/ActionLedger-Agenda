# # src/tools/mongo_toolkit.py
# from typing import List, Optional, Dict
# from dotenv import dotenv_values
# from langchain_core.tools import BaseTool
# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
# from langchain_mongodb.agent_toolkit.database import MongoDBDatabase

# # Simple connection cache so multiple agents can share the same handle
# _CONN_CACHE: Dict[str, MongoDBDatabase] = {}


# def get_mongo_tools(
#     llm: BaseChatModel,
#     uri: Optional[str] = None,
#     database: Optional[str] = None,
# ) -> List[BaseTool]:
#     """
#     Return a reusable list of MongoDB LangChain tools bound to the given LLM and DB.

#     Priority for config:
#       1) explicit args (uri, database)
#       2) .env keys: MONGO_CLIENT / MONGODB_URI, MONGO_DB / MONGODB_DB

#     Raises:
#       ValueError if URI/DB are not provided via args or env.
#     """
#     env = dotenv_values()
#     uri = uri or env.get("MONGO_CLIENT") or env.get("MONGODB_URI")
#     database = database or env.get("MONGO_DB") or env.get("MONGODB_DB")

#     if not uri or not database:
#         raise ValueError(
#             "MongoDB URI/DB required. Pass uri/database or set "
#             "MONGO_CLIENT/MONGODB_URI and MONGO_DB/MONGODB_DB in the environment."
#         )

#     cache_key = f"{uri}::{database}"
#     db = _CONN_CACHE.get(cache_key)
#     if db is None:
#         db = MongoDBDatabase.from_connection_string(uri, database=database)
#         _CONN_CACHE[cache_key] = db

#     toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
#     return toolkit.get_tools()


# def close_all_mongo_connections() -> None:
#     """Optional: clear cached DB wrappers (for app shutdown or test teardown)."""
#     _CONN_CACHE.clear()


# # src/tools/mongo_toolkit.py
# from typing import List, Optional, Dict
# from dotenv import dotenv_values
# from langchain_core.tools import BaseTool
# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
# from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
# from pymongo import MongoClient  # <— add this

# _CONN_CACHE: Dict[str, MongoDBDatabase] = {}


# def _get_env_uri_db(uri: Optional[str], database: Optional[str]) -> tuple[str, str]:
#     env = dotenv_values()
#     uri = uri or env.get("MONGO_CLIENT") or env.get("MONGODB_URI")
#     database = database or env.get("MONGO_DB") or env.get("MONGODB_DB")
#     if not uri or not database:
#         raise ValueError(
#             "MongoDB URI/DB required. Pass uri/database or set "
#             "MONGO_CLIENT/MONGODB_URI and MONGO_DB/MONGODB_DB in the environment."
#         )
#     return uri, database


# def _get_or_create_db(uri: str, database: str) -> MongoDBDatabase:
#     cache_key = f"{uri}::{database}"
#     if cache_key in _CONN_CACHE:
#         return _CONN_CACHE[cache_key]

#     # 1) Ping using official MongoClient (no private attrs)
#     try:
#         with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
#             client.admin.command("ping")
#             _ = client[database].list_collection_names()
#     except Exception as e:
#         raise ValueError(f"Failed to connect to MongoDB database '{database}': {e}") from e

#     # 2) Wrap with LangChain’s MongoDBDatabase
#     db = MongoDBDatabase.from_connection_string(uri, database=database)
#     _CONN_CACHE[cache_key] = db
#     return db


# def get_mongo_tools(
#     llm: BaseChatModel,
#     uri: Optional[str] = None,
#     database: Optional[str] = None,
# ) -> List[BaseTool]:
#     uri, database = _get_env_uri_db(uri, database)
#     db = _get_or_create_db(uri, database)
#     toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
#     return toolkit.get_tools()


# def close_all_mongo_connections() -> None:
#     _CONN_CACHE.clear()

# src/tools/mongo_toolkit.py
import ast
import json
from typing import List, Optional, Dict, Any

from dotenv import dotenv_values
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from pydantic.v1 import BaseModel, Field # Use Pydantic for schema definition
from pymongo import MongoClient

_CONN_CACHE: Dict[str, MongoDBDatabase] = {}

_ALLOWED_TOOL_NAMES = {
    # keep just what you need
    "mongodb_query",
    # if you use your own wrapper:
    "custom_mongodb_query",
    # comment out if you don't want the model to introspect schema/collections:
    # "mongodb_schema",
}

def _get_env_uri_db(uri: Optional[str], database: Optional[str]) -> tuple[str, str]:
    env = dotenv_values()
    uri = uri or env.get("MONGO_CLIENT") or env.get("MONGODB_URI")
    database = database or env.get("MONGO_DB") or env.get("MONGODB_DB")
    if not uri or not database:
        raise ValueError(
            "MongoDB URI/DB required. Pass uri/database or set "
            "MONGO_CLIENT/MONGODB_URI and MONGO_DB/MONGODB_DB in the environment."
        )
    return uri, database


def _get_or_create_db(uri: str, database: str) -> MongoDBDatabase:
    cache_key = f"{uri}::{database}"
    if cache_key in _CONN_CACHE:
        return _CONN_CACHE[cache_key]

    # 1) Ping using official MongoClient (no private attrs)
    try:
        with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
            client.admin.command("ping")
            _ = client[database].list_collection_names()
    except Exception as e:
        raise ValueError(f"Failed to connect to MongoDB database '{database}': {e}") from e

    # 2) Wrap with LangChain’s MongoDBDatabase
    db = MongoDBDatabase.from_connection_string(uri, database=database)
    _CONN_CACHE[cache_key] = db
    return db

# 1. Define the input schema for our custom tool using Pydantic
class MongoQueryInput(BaseModel):
    collection: str = Field(description="The name of the MongoDB collection to query.")
    query: Dict[str, Any] = Field(
        description="The MongoDB query filter to apply, as a dictionary. Example: {'_id': 'some_id'}"
    )

# 2. Create the custom, robust tool
@tool(args_schema=MongoQueryInput)
def custom_mongodb_query(collection: str, query: Dict[str, Any]) -> str:
    """
    Executes a 'find' query on a specified MongoDB collection using a filter.
    Use this to fetch specific documents from the database.
    """
    try:
        # NOTE: No need for ast.literal_eval here because Pydantic + LangChain handle it!
        # The schema definition ensures we get the right data types.

        uri, database = _get_env_uri_db(None, None)
        db_interface = _get_or_create_db(uri, database)
        
        # Use the underlying pymongo client for the actual query
        results = list(db_interface._db[collection].find(query))

        # Sanitize ObjectId for JSON serialization
        for doc in results:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
        if not results:
            return f"No documents found in collection '{collection}' for the query: {query}"
            
        return json.dumps(results, indent=2)

    except Exception as e:
        return f"Error: Failed to execute query on collection '{collection}'. Detail: {e}. Please check your collection name and query syntax."


def get_mongo_tools(
    llm: BaseChatModel,
    uri: Optional[str] = None,
    database: Optional[str] = None,
) -> List[BaseTool]:
    """
    Gets the MongoDB tools. Includes a custom, robust query tool
    and standard tools for listing collections and getting schemas.
    """
    uri, database = _get_env_uri_db(uri, database)
    db = _get_or_create_db(uri, database)
    toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
    
    # Get standard tools BUT filter out the default query tool
    standard_tools = [
        t for t in toolkit.get_tools() if t.name != "mongodb_query"
    ]
    
    # Add our custom tool instead. Let's rename it for clarity in logs if we want.
    # custom_mongodb_query.name = "mongodb_query" # Keep the original name
    
    all_tools = standard_tools + [custom_mongodb_query]
    filtered = [t for t in all_tools if t.name in _ALLOWED_TOOL_NAMES]
    return filtered


def close_all_mongo_connections() -> None:
    _CONN_CACHE.clear()
