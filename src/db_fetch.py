import pymongo
import json
import asyncio
from dotenv import dotenv_values


# ---------- FETCH ALL COLLECTIONS ----------
async def fetch_all_from_mongo(mongo_client: str, mongo_db: str):
    client = pymongo.MongoClient(mongo_client)
    db = client[mongo_db]

    collections = db.list_collection_names()
    all_data = {}

    for coll_name in collections:
        collection = db[coll_name]
        docs = list(collection.find({}, {"_id": 0}))  # exclude _id for readability
        all_data[coll_name] = docs

    client.close()
    print(json.dumps(all_data, indent=2))
    return all_data


# ---------- FETCH SPECIFIC COLLECTION ----------
def fetch_from_collection(mongo_client: str, mongo_db: str, collection_name: str):
    client = pymongo.MongoClient(mongo_client)
    db = client[mongo_db]
    docs = list(db[collection_name].find({}, {"_id": 0}))
    client.close()
    return docs


def read_start():
    env_vals = dotenv_values()
    mongo_client = env_vals["MONGO_CLIENT"]
    mongo_db = env_vals["MONGO_DB"]

    asyncio.run(fetch_all_from_mongo(mongo_client, mongo_db))

    summaries = fetch_from_collection(mongo_client, mongo_db, "summary")
    print("Summaries:\n", json.dumps(summaries, indent=2))


if __name__ == "__main__":
    read_start()
