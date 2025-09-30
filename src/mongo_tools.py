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

# # Running mongo db tool with open ai
# import ast
# import json
# from typing import List, Optional, Dict, Any

# from dotenv import dotenv_values
# from langchain_core.tools import BaseTool, tool
# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
# from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
# from pydantic.v1 import BaseModel, Field # Use Pydantic for schema definition
# from pymongo import MongoClient

# _CONN_CACHE: Dict[str, MongoDBDatabase] = {}

# _ALLOWED_TOOL_NAMES = {
#     # keep just what you need
#     "mongodb_query",
#     # if you use your own wrapper:
#     "custom_mongodb_query",
#     # comment out if you don't want the model to introspect schema/collections:
#     # "mongodb_schema",
# }

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

# # 1. Define the input schema for our custom tool using Pydantic
# class MongoQueryInput(BaseModel):
#     collection: str = Field(description="The name of the MongoDB collection to query.")
#     query: Dict[str, Any] = Field(
#         description="The MongoDB query filter to apply, as a dictionary. Example: {'_id': 'some_id'}"
#     )

# # 2. Create the custom, robust tool
# @tool(args_schema=MongoQueryInput)
# def custom_mongodb_query(collection: str, query: Dict[str, Any]) -> str:
#     """
#     Executes a 'find' query on a specified MongoDB collection using a filter.
#     Use this to fetch specific documents from the database.
#     """
#     try:
#         # NOTE: No need for ast.literal_eval here because Pydantic + LangChain handle it!
#         # The schema definition ensures we get the right data types.

#         uri, database = _get_env_uri_db(None, None)
#         db_interface = _get_or_create_db(uri, database)
        
#         # Use the underlying pymongo client for the actual query
#         results = list(db_interface._db[collection].find(query))

#         # Sanitize ObjectId for JSON serialization
#         for doc in results:
#             if '_id' in doc:
#                 doc['_id'] = str(doc['_id'])
        
#         if not results:
#             return f"No documents found in collection '{collection}' for the query: {query}"
            
#         return json.dumps(results, indent=2)

#     except Exception as e:
#         return f"Error: Failed to execute query on collection '{collection}'. Detail: {e}. Please check your collection name and query syntax."


# def get_mongo_tools(
#     llm: BaseChatModel,
#     uri: Optional[str] = None,
#     database: Optional[str] = None,
# ) -> List[BaseTool]:
#     """
#     Gets the MongoDB tools. Includes a custom, robust query tool
#     and standard tools for listing collections and getting schemas.
#     """
#     uri, database = _get_env_uri_db(uri, database)
#     db = _get_or_create_db(uri, database)
#     toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
    
#     # Get standard tools BUT filter out the default query tool
#     standard_tools = [
#         t for t in toolkit.get_tools() if t.name != "mongodb_query"
#     ]
    
#     # Add our custom tool instead. Let's rename it for clarity in logs if we want.
#     # custom_mongodb_query.name = "mongodb_query" # Keep the original name
    
#     all_tools = standard_tools + [custom_mongodb_query]
#     filtered = [t for t in all_tools if t.name in _ALLOWED_TOOL_NAMES]
#     return filtered


# def close_all_mongo_connections() -> None:
#     _CONN_CACHE.clear()


# try with gemini
import json
import re
from typing import List, Optional, Dict, Any, Literal, Union

from dotenv import dotenv_values
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from pydantic.v1 import BaseModel, Field, ValidationError
from pymongo import MongoClient
from bson import json_util  # for safe BSON -> JSON (ObjectId, dates, etc.)

_CONN_CACHE: Dict[str, MongoDBDatabase] = {}

_ALLOWED_TOOL_NAMES = {
    # expose only what you want the model to use
    "custom_mongodb_query",
    "mongodb_query_checker",
    # uncomment if you also want the LLM to inspect collections
    # "mongodb_list_collections",
    # keep default query tool out to avoid duplicate entry points
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

    # 1) Ping using official MongoClient
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


# # ---------- Coercion & validation helpers ----------
# def _coerce_query_to_dict(q: Union[Dict[str, Any], str]) -> Dict[str, Any]:
#     """
#     Accepts dict or JSON-string; returns dict.
#     Handles common bad cases from some models:
#       - Backslash-escaped JSON: "{\"_id\":\"thread_7\"}"
#       - Single quotes: '{\'_id\': \"thread_7\"}'
#       - Wrapped as JSON string literal => json.loads twice
#     """
#     if isinstance(q, dict):
#         return q

#     if not isinstance(q, str):
#         raise ValueError("query must be a dict or JSON string")

#     s = q.strip()

#     # If it's a JSON string literal like "\"{...}\"", unquote once
#     if s.startswith('"') and s.endswith('"'):
#         try:
#             s = json.loads(s)
#         except Exception:
#             # if it wasn't actually a JSON string, continue with s as-is
#             pass
#         if isinstance(s, str):
#             s = s.strip()

#     # Heuristic: replace lone single quotes with double quotes if no doubles exist
#     if "'" in s and '"' not in s:
#         s = s.replace("'", '"')

#     # Remove superfluous backslashes before quotes
#     s = re.sub(r'\\+"', '"', s)

#     try:
#         return json.loads(s)
#     except Exception as e:
#         raise ValueError(f"query is not valid JSON after coercion: {e}")


def _coerce_query_to_dict(q: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    if isinstance(q, dict):
        return q
    if not isinstance(q, str):
        raise ValueError("query must be a dict or JSON string")

    s = q.strip()

    # If it's \"{...}\" (a JSON string literal containing JSON), unquote once
    if s.startswith('"') and s.endswith('"'):
        try:
            s = json.loads(s)
        except Exception:
            pass
        if isinstance(s, str):
            s = s.strip()

    # Fast path: already looks like JSON object
    if s.startswith('{') and s.endswith('}'):
        try:
            return json.loads(s)
        except Exception:
            pass  # fallthrough to gentle fixes

    # Heuristic: single quotes only → switch to double quotes
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')

    # Remove escaped quotes only (\" -> ")
    s = re.sub(r'\\"', '"', s)

    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"query is not valid JSON after coercion: {e}")


def _validate_allowed_shape(collection: str, qdict: Dict[str, Any]) -> tuple[bool, str]:
    """
    Enforce your strict patterns (adjust as needed):
      - For 'cv' and 'summary': {'_id': '<thread_id>'}
      - For 'question_guidelines':
            {'_id': {'$in': [...]}} OR {'_id': {'$regex': ...}}
    """
    if collection in ("cv", "summary"):
        if not isinstance(qdict, dict) or qdict.get("_id") is None:
            return False, "For 'cv' and 'summary', query must be {'_id': '<thread_id>'}."
        if not isinstance(qdict["_id"], str):
            return False, "The '_id' value must be a string."
        return True, "ok"

    if collection == "question_guidelines":
        if not isinstance(qdict, dict) or "_id" not in qdict:
            return False, "For 'question_guidelines', query must include '_id'."
        _id = qdict["_id"]
        if isinstance(_id, dict):
            if "$in" in _id:
                if not isinstance(_id["$in"], list) or not all(isinstance(x, str) for x in _id["$in"]):
                    return False, "'$in' must be a list of strings."
                return True, "ok"
            if "$regex" in _id:
                if not isinstance(_id["$regex"], str):
                    return False, "'$regex' must be a string."
                return True, "ok"
            return False, "Only '$in' or '$regex' supported for 'question_guidelines' _id."
        elif isinstance(_id, str):
            return True, "ok"
        else:
            return False, "'_id' must be a string or an object with '$in'/'$regex'."

    return False, f"Collection '{collection}' is not allowed."


# ---------- Tools ----------
class MongoQueryInput(BaseModel):
    collection: Literal["cv", "summary", "question_guidelines"] = Field(
        description="Target MongoDB collection."
    )
    # accept dict or string; we'll coerce internally
    query: Union[Dict[str, Any], str] = Field(
        description="Mongo filter as dict or JSON string. Example: {'_id': 'thread_7'}"
    )


@tool("mongodb_query_checker", args_schema=MongoQueryInput)
def mongodb_query_checker(collection: str, query: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Validate and normalize a MongoDB query before execution.
    Returns: { valid: bool, reason?: str, normalized_query?: dict }
    """
    try:
        qdict = _coerce_query_to_dict(query)
    except ValueError as e:
        return {"valid": False, "reason": str(e)}

    ok, reason = _validate_allowed_shape(collection, qdict)
    if not ok:
        return {"valid": False, "reason": reason}

    return {"valid": True, "normalized_query": qdict}


@tool("custom_mongodb_query", args_schema=MongoQueryInput)
def custom_mongodb_query(collection: str, query: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Execute a sanitized 'find' query on the specified collection.
    Always call `mongodb_query_checker` first and then pass its `normalized_query` here.
    Returns: { ok: bool, data?: [...], count?: int, error?: str }
    """
    try:
        # Normalize again in case the model skipped the checker
        qdict = _coerce_query_to_dict(query)
        ok, reason = _validate_allowed_shape(collection, qdict)
        if not ok:
            return {"ok": False, "error": f"Invalid query shape: {reason}"}

        uri, database = _get_env_uri_db(None, None)
        db_interface = _get_or_create_db(uri, database)

        # Perform the query
        docs = list(db_interface._db[collection].find(qdict))

        # Safe JSON via bson.json_util; then back to Python obj for uniform return
        json_text = json_util.dumps(docs)           # handles ObjectId, datetime, etc.
        data = json.loads(json_text)

        return {"ok": True, "count": len(data), "data": data}

    except Exception as e:
        return {"ok": False, "error": f"Failed to execute query on '{collection}': {e}"}


# Optional: list collections tool (expose only if you want the LLM to use it)
class ListCollectionsInput(BaseModel):
    pass

@tool("mongodb_list_collections", args_schema=ListCollectionsInput)
def mongodb_list_collections() -> Dict[str, Any]:
    """
    List collection names (optional tool; include in _ALLOWED_TOOL_NAMES if desired).
    """
    try:
        uri, database = _get_env_uri_db(None, None)
        db_interface = _get_or_create_db(uri, database)
        names = db_interface._db.list_collection_names()
        return {"ok": True, "collections": names}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_mongo_tools(
    llm: BaseChatModel,
    uri: Optional[str] = None,
    database: Optional[str] = None,
) -> List[BaseTool]:
    """
    Return curated Mongo tools. We filter to only allowed tools by name.
    """
    uri, database = _get_env_uri_db(uri, database)
    _ = _get_or_create_db(uri, database)  # ensure connection upfront

    # You *can* still build the default toolkit for future use, but we’ll not expose its query tool
    _ = MongoDBDatabaseToolkit(db=MongoDBDatabase.from_connection_string(uri, database), llm=llm)

    # Build our final tool list
    all_tools: List[BaseTool] = [
        custom_mongodb_query,
        mongodb_query_checker,
        # mongodb_list_collections,  # include if needed; also add to _ALLOWED_TOOL_NAMES
    ]

    # Only keep whitelisted tools
    filtered = [t for t in all_tools if t.name in _ALLOWED_TOOL_NAMES]
    return filtered


def close_all_mongo_connections() -> None:
    _CONN_CACHE.clear()
