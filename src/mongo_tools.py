import json
import re
from typing import List, Optional, Dict, Any, Literal, Union

from dotenv import dotenv_values
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from pydantic.v1 import BaseModel, Field
from pymongo import MongoClient
from bson import json_util


# In-process cache of DB interfaces keyed by "<uri>::<database>"
_CONN_CACHE: Dict[str, MongoDBDatabase] = {}

# Only these tool names will be returned by get_mongo_tools()
_ALLOWED_TOOL_NAMES = {
    "custom_mongodb_query",
    "mongodb_query_checker",
}


def _get_env_uri_db(uri: Optional[str], database: Optional[str]) -> tuple[str, str]:
    """
    Resolve MongoDB URI and database:
      - Prefer explicit function arguments
      - Fallback to environment (.env) keys: MONGO_CLIENT/MONGODB_URI and MONGO_DB/MONGODB_DB
    """
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
    """
    Return a MongoDBDatabase (langchain-mongodb) instance, creating and caching it if needed.
    Performs a quick 'ping' and lists collections to validate connectivity.
    """
    cache_key = f"{uri}::{database}"
    if cache_key in _CONN_CACHE:
        return _CONN_CACHE[cache_key]

    try:
        # Lightweight connectivity check before constructing toolkit DB wrapper
        with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
            client.admin.command("ping")
            _ = client[database].list_collection_names()
    except Exception as e:
        raise ValueError(f"Failed to connect to MongoDB database '{database}': {e}") from e

    db = MongoDBDatabase.from_connection_string(uri, database=database)
    _CONN_CACHE[cache_key] = db
    return db


def _coerce_query_to_dict(q: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Accept either a dict or a JSON string and return a dict.
    Handles common pitfalls: extra quoting, single quotes, and escaped quotes.
    """
    if isinstance(q, dict):
        return q
    if not isinstance(q, str):
        raise ValueError("query must be a dict or JSON string")

    s = q.strip()

    # If the entire JSON is a string literal like "\"{...}\"", unquote once.
    if s.startswith('"') and s.endswith('"'):
        try:
            s = json.loads(s)
        except Exception:
            pass
        if isinstance(s, str):
            s = s.strip()

    # Fast path: looks like a JSON object
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            # fall through to gentle fixes
            pass

    # Heuristic: only single quotes present → switch to double quotes
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')

    # Remove escaped quotes (\" -> ")
    s = re.sub(r'\\"', '"', s)

    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"query is not valid JSON after coercion: {e}")


def _validate_allowed_shape(collection: str, qdict: Dict[str, Any]) -> tuple[bool, str]:
    """
    Enforce strict, collection-specific query shapes to reduce risk:
      - 'cv' and 'summary': {'_id': '<thread_id>'} with _id as string
      - 'question_guidelines': '_id' supports:
          * string
          * {'$in': [str, ...]}
          * {'$regex': str}
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


# ------------------------------ Tool schemas ------------------------------ #
class MongoQueryInput(BaseModel):
    """Arguments for Mongo query tools."""
    collection: Literal["cv", "summary", "question_guidelines"] = Field(
        description="Target MongoDB collection."
    )
    query: Union[Dict[str, Any], str] = Field(
        description="Mongo filter as dict or JSON string. Example: {'_id': 'thread_7'}"
    )


# ------------------------------ Tools ------------------------------ #
@tool("mongodb_query_checker", args_schema=MongoQueryInput)
def mongodb_query_checker(collection: str, query: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Validate and normalize a MongoDB query before execution.
    Returns a dict with:
      - valid: bool
      - reason: str (when valid == False)
      - normalized_query: dict (when valid == True)
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
    Expected call pattern: run `mongodb_query_checker` first and pass its normalized_query here.

    Returns:
      { ok: bool, data?: list, count?: int, error?: str }
    """
    try:
        # Normalize again in case the caller skipped the checker
        qdict = _coerce_query_to_dict(query)
        ok, reason = _validate_allowed_shape(collection, qdict)
        if not ok:
            return {"ok": False, "error": f"Invalid query shape: {reason}"}

        uri, database = _get_env_uri_db(None, None)
        db_interface = _get_or_create_db(uri, database)

        # Perform the query
        docs = list(db_interface._db[collection].find(qdict))

        # BSON-safe serialization then back to Python for consistent JSON output
        json_text = json_util.dumps(docs)
        data = json.loads(json_text)

        return {"ok": True, "count": len(data), "data": data}

    except Exception as e:
        return {"ok": False, "error": f"Failed to execute query on '{collection}': {e}"}


class ListCollectionsInput(BaseModel):
    """No arguments."""
    pass


@tool("mongodb_list_collections", args_schema=ListCollectionsInput)
def mongodb_list_collections() -> Dict[str, Any]:
    """
    Optionally list collection names. Not included by default in the exported toolset.
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
    Build and return the curated list of Mongo tools, filtered by _ALLOWED_TOOL_NAMES.
    Establishes a connection up front to fail fast if credentials are invalid.
    """
    uri, database = _get_env_uri_db(uri, database)
    _ = _get_or_create_db(uri, database)  # ensure connectivity

    # Initialize the toolkit (kept for side effects/integration patterns)
    _ = MongoDBDatabaseToolkit(
        db=MongoDBDatabase.from_connection_string(uri, database),
        llm=llm,
    )

    all_tools: List[BaseTool] = [
        custom_mongodb_query,
        mongodb_query_checker,
        # mongodb_list_collections,  # enable if desired and add to _ALLOWED_TOOL_NAMES
    ]

    return [t for t in all_tools if t.name in _ALLOWED_TOOL_NAMES]


def close_all_mongo_connections() -> None:
    """Clear the in-process DB interface cache."""
    _CONN_CACHE.clear()
