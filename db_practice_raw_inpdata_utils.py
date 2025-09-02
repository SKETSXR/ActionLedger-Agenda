import os
import pymongo
import pprint
import json
import yaml
from typing import Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_community.document_loaders import JSONLoader
from dotenv import dotenv_values
from langchain.schema import Document
import hashlib
from pymongo import ASCENDING

def load_config(path: str) -> dict[str, Any] | list[Any]:
    with open(path) as f:
        config = yaml.safe_load(f)

    return config

# def inp_data_storage(inp_cv: str, inp_jd: str, inp_skill_tree: str):
#     env_vals = dotenv_values()
#     os.environ["OPENAI_API_KEY"] = env_vals['OPENAI_API_KEY']
#     MONGO_CONNECTION_STRING = env_vals['MONGO_CLIENT']
#     MONGO_DB = env_vals['MONGO_DB']


#     # connect to Mongo (local server, default port 27017)
#     client = MongoClient(MONGO_CONNECTION_STRING)

#     # pick a database and collection
#     db = client[MONGO_DB]
#     collection = db["jd"]
#     result = collection.insert_one(json.loads(inp_jd))

#     # print("Inserted ID:", result.inserted_id)
#     col = db["cv"]
#     doc = json.loads(inp_cv)
#     col.insert_one({k: doc[k] for k in ("projects","experience")})
#     col = db["skill_tree"]
#     doc = json.loads(inp_skill_tree)
#     result = col.insert_one(json.loads(inp_skill_tree))



def canonical_hash(doc: dict) -> str:
    s = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def inp_data_storage(inp_cv: str, inp_jd: str, inp_skill_tree: str):
    env_vals = dotenv_values()
    os.environ["OPENAI_API_KEY"] = env_vals['OPENAI_API_KEY']
    MONGO_CONNECTION_STRING = env_vals['MONGO_CLIENT']
    MONGO_DB = env_vals['MONGO_DB']


    # connect to Mongo (local server, default port 27017)
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[MONGO_DB]

    # --- JD ---
    jd_col = db["jd"]
    jd_col.create_index([("doc_hash", ASCENDING)], unique=True)

    jd = json.loads(inp_jd)
    jd_hash = canonical_hash(jd)

    res = jd_col.update_one(
        {"doc_hash": jd_hash},
        {"$setOnInsert": {**jd, "doc_hash": jd_hash}},
        upsert=True
    )
    print("JD inserted" if res.upserted_id else "JD duplicate, skipped")

    # --- CV (only projects+experience) ---
    cv_col = db["cv"]
    cv_doc = json.loads(inp_cv)
    cv_subset = {k: cv_doc[k] for k in ("projects", "experience") if k in cv_doc}
    cv_hash = canonical_hash(cv_subset)
    cv_col.create_index([("doc_hash", ASCENDING)], unique=True)

    res = cv_col.update_one(
        {"doc_hash": cv_hash},
        {"$setOnInsert": {**cv_subset, "doc_hash": cv_hash}},
        upsert=True
    )
    print("CV inserted" if res.upserted_id else "CV duplicate, skipped")

    # --- Skill Tree ---
    st_col = db["skill_tree"]
    st = json.loads(inp_skill_tree)
    st_hash = canonical_hash(st)
    st_col.create_index([("doc_hash", ASCENDING)], unique=True)

    res = st_col.update_one(
        {"doc_hash": st_hash},
        {"$setOnInsert": {**st, "doc_hash": st_hash}},
        upsert=True
    )
    print("Skill tree inserted" if res.upserted_id else "Skill tree duplicate, skipped")
