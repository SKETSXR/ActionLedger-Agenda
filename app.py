# =============================================================================
# Script: app.py
# =============================================================================
# Purpose
#   Load inputs (JD, CV and Skill Tree from a mongo db database and Question Guidelines from a JSON file),
#   builds the pipeline InputSchema, executes the AgendaGenerationAgent graph,
#   prints artifacts, and write a combined output file.
#
# Behavior
#   - Uses a parse functions to build Pydantic models from fetched dictionaries
#   - Loads question guidelines from a JSON file
#   - Prints the same debug parsed inputs before running the graph
#   - Invokes the compiled graph with the loaded config
#   - Logs each output section and writes a combined text file
#   - Reports total time in minutes
# =============================================================================

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.agent.AgendaGenerationAgent import run_agenda_with_logging
from src.schema.input_schema import (
    CandidateProfileSchema,
    InputSchema,
    JobDescriptionSchema,
    SkillTreeSchema,
)


# ---------------------------------------------------------------------
# Mongo helpers
# ---------------------------------------------------------------------
def _mongo_client() -> MongoClient:
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI is not set in environment")
    return MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)


def _get_db(client: MongoClient):
    db_name = os.environ.get("MONGO_DB2", "agenda_inputs")
    return client[db_name]


def _fetch_by_id(db, collection: str, doc_id: int) -> Dict[str, Any]:
    """
    Fetch a document by integer 'id' from the given collection.
    Returns the document without Mongo's '_id'.
    Raises KeyError if not found.
    """
    doc = db[collection].find_one({"_id": int(doc_id)}, {"_id": 0})
    if not doc:
        raise KeyError(f"Document with id={doc_id} not found in '{collection}'")
    return doc


# ---------------------------------------------------------------------
# Local I/O helpers
# ---------------------------------------------------------------------
def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _skill_tree_from_dict(tree_dict: dict) -> SkillTreeSchema:
    """
    Build SkillTreeSchema from dict.
    If '_id' is present (for storage convenience), drop it because
    the Pydantic schema likely doesn't expect it.
    """
    tree = dict(tree_dict)
    tree.pop("_id", None)
    return SkillTreeSchema(**tree)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    start_ns = time.time_ns()

    # ---- IDs to fetch (ints). Set via env; defaults shown here. ----
    JD_ID = int(os.environ.get("JD_ID", 3))
    CV_ID = int(os.environ.get("CV_ID", 3))
    ST_ID = int(os.environ.get("ST_ID", 3))

    # Keep question guidelines from file
    qg_json_path = Path(
        os.environ.get(
            "QG_JSON_PATH", r"testing\custom_testing_inputs\question_guidelines2.json"
        )
    )
    config_yaml_path = Path(os.environ.get("CONFIG_YAML_PATH", "config.yaml"))
    output_txt_path = Path(os.environ.get("OUTPUT_TXT_PATH", r"testing\op24.txt"))

    try:
        client = _mongo_client()
        client.admin.command("ping")
        db = _get_db(client)

        # NOTE: collections are EXACT names you requested
        jd_doc = _fetch_by_id(db, "jd", JD_ID)
        cv_doc = _fetch_by_id(db, "cv", CV_ID)
        st_doc = _fetch_by_id(db, "skill_tree", ST_ID)

    except (PyMongoError, KeyError, RuntimeError) as e:
        raise SystemExit(f"[Mongo Error] {e}")

    # ---- Build Pydantic models ----
    # Drop '_id' fields if present; schemas likely don't take them.
    jdes = JobDescriptionSchema(
        job_role=jd_doc["job_role"],
        company_background=jd_doc["company_background"],
        fundamental_knowledge=jd_doc.get("fundamental_knowledge"),
    )

    cp = CandidateProfileSchema(
        skills=cv_doc["skills"],
        projects=cv_doc["projects"],
        experience=cv_doc["experience"],
    )

    skill_tree = _skill_tree_from_dict(st_doc)

    # Question guidelines: plain dict from file
    with qg_json_path.open("r", encoding="utf-8") as f:
        question_guidelines = json.load(f)

    # ---- Build InputSchema and run your graph ----
    inp = InputSchema(
        job_description=jdes,
        skill_tree=skill_tree,
        candidate_profile=cp,
        question_guidelines=question_guidelines,
    )

    # Debug prints (preserved)
    print(inp.candidate_profile.model_dump_json(indent=2) + "\n")
    print(inp.job_description.model_dump_json(indent=2) + "\n")
    print(inp.skill_tree.model_dump_json(indent=2))

    config = _read_yaml(config_yaml_path)
    otpt = asyncio.run(run_agenda_with_logging(inp, config))

    # Collect & save outputs
    combined_text = ""
    for k, v in otpt.items():
        section = str(k).capitalize()
        body = v.model_dump_json(indent=2)
        print(f"\n{section} --->\n\n {body}\n")
        combined_text += f"\n{section} --->\n\n {body}\n"

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_txt_path.open("w", encoding="utf-8") as f:
        f.write(combined_text)

    end_ns = time.time_ns()
    print(f"\nTime taken: {(end_ns - start_ns) / 60000000000} mins")
