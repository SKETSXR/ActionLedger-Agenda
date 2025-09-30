
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

_CONN_CACHE: Dict[str, MongoDBDatabase] = {}
_ALLOWED_TOOL_NAMES = {
    "custom_mongodb_query",
    "mongodb_query_checker",
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

    try:
        with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
            client.admin.command("ping")
            _ = client[database].list_collection_names()
    except Exception as e:
        raise ValueError(f"Failed to connect to MongoDB database '{database}': {e}") from e

    db = MongoDBDatabase.from_connection_string(uri, database=database)
    _CONN_CACHE[cache_key] = db
    return db


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


# Optional: list collections tool
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
