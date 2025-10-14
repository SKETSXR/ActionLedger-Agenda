# # =============================================================================
# # Script: app.py
# # =============================================================================
# # Purpose
# #   Load test inputs (JD text, CV PDF, Skill Tree JSON, Question Guidelines),
# #   parse JD/CV via existing tools, build the pipeline InputSchema, execute the
# #   AgendaGenerationAgent graph, print artifacts, and write a combined output file.
# #
# # Behavior
# #   - Reads from the same hardcoded paths
# #   - Uses the same parse functions and error messages
# #   - Prints the same debug dumps before running the graph
# #   - Invokes the compiled graph with the loaded config
# #   - Prints each output section and writes a combined text file
# #   - Reports total time in minutes
# # =============================================================================

# import asyncio
# import json
# import time
# import yaml

# from pathlib import Path

# from dotenv import load_dotenv

# from src.agent.AgendaGenerationAgent import AgendaGenerationAgent
# from src.schema.input_schema import (
#     JobDescriptionSchema,
#     CandidateProfileSchema,
#     SkillTreeSchema,
#     InputSchema,
# )
# from src.tools.jd_extraction import parse_jd_text_to_json
# from src.tools.cv_extraction import parse_pdf_to_json


# # ----------------------------- Helpers ----------------------------- #

# def _read_text(path: Path) -> str:
#     with path.open("r", encoding="utf-8") as f:
#         return f.read()


# def _read_json(path: Path) -> dict:
#     with path.open("r", encoding="utf-8") as f:
#         return json.load(f)


# def _read_yaml(path: Path) -> dict:
#     with path.open("r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def _skill_tree_from_dict(tree_dict: dict) -> SkillTreeSchema:
#     # Construct SkillTreeSchema from a JSON dict (unchanged behavior)
#     return SkillTreeSchema(**tree_dict)


# # ----------------------------- Main ----------------------------- #

# if __name__ == "__main__":
#     start_ns = time.time_ns()
#     load_dotenv()

#     # Paths (kept identical semantics; normalized through Path)
#     jd_txt_path = Path(r"testing\Ayam\jd.txt")
#     cv_pdf_path = Path(r"testing\Ayam\AyamHeniberMeitei_2025L - ayam heniber.pdf")
#     skill_tree_json_path = Path(r"testing\custom_testing_inputs\skilltree3_priority.json")
#     qg_json_path = Path(r"testing\custom_testing_inputs\question_guidelines1.json")
#     config_yaml_path = Path("config.yaml")
#     output_txt_path = Path(r"testing\op7.txt")

#     # --- JD parse ---
#     jd_text = _read_text(jd_txt_path)
#     jd_json_str = asyncio.run(parse_jd_text_to_json(jd_text))
#     if jd_json_str == "JD not contain any text":
#         raise RuntimeError("Open AI API not running")

#     jd_obj = json.loads(jd_json_str)
#     jdes = JobDescriptionSchema(
#         job_role=jd_obj["job_role"],
#         company_background=jd_obj["company_background"],
#         fundamental_knowledge=jd_obj.get("fundamental_knowledge"),
#     )

#     # --- CV parse ---
#     cv_json_str = asyncio.run(parse_pdf_to_json(str(cv_pdf_path)))
#     if cv_json_str == "CV does not contain proper text":
#         raise RuntimeError("Open AI API not running")
#     cv_obj = json.loads(cv_json_str)

#     cp = CandidateProfileSchema(
#         skills=cv_obj["skills"],
#         projects=cv_obj["projects"],
#         experience=cv_obj["experience"],
#     )

#     # --- Skill tree ---
#     skill_tree_dict = _read_json(skill_tree_json_path)
#     skill_tree = _skill_tree_from_dict(skill_tree_dict)

#     # --- Question guidelines ---
#     question_guidelines = _read_json(qg_json_path)

#     # Build the input schema for the pipeline
#     inp = InputSchema(
#         job_description=jdes,
#         skill_tree=skill_tree,
#         candidate_profile=cp,
#         question_guidelines=question_guidelines,
#     )

#     # Debug prints (preserved)
#     print(inp.candidate_profile.model_dump_json(indent=2) + "\n")
#     print(inp.job_description.model_dump_json(indent=2) + "\n")
#     print(inp.skill_tree.model_dump_json(indent=2))

#     # Load config and run the graph
#     config = _read_yaml(config_yaml_path)
#     graph = AgendaGenerationAgent.get_graph()
#     otpt = asyncio.run(graph.ainvoke(inp, config))

#     # Collect & save outputs
#     combined_text = ""
#     for k, v in otpt.items():
#         section = str(k).capitalize()
#         body = v.model_dump_json(indent=2)
#         print(f"\n{section} --->\n\n {body}\n")
#         combined_text += f"\n{section} --->\n\n {body}\n"

#     output_txt_path.parent.mkdir(parents=True, exist_ok=True)
#     with output_txt_path.open("w", encoding="utf-8") as f:
#         f.write(combined_text)

#     end_ns = time.time_ns()
#     print(f"\nTime taken: {(end_ns - start_ns) / 60000000000} mins")


import asyncio
import json
import os
import time
import yaml
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.agent.AgendaGenerationAgent import AgendaGenerationAgent
from src.schema.input_schema import (
    JobDescriptionSchema,
    CandidateProfileSchema,
    SkillTreeSchema,
    InputSchema,
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
    JD_ID = int(os.environ.get("JD_ID", 1))
    CV_ID = int(os.environ.get("CV_ID", 1))
    ST_ID = int(os.environ.get("ST_ID", 1))

    # Keep question guidelines from file
    qg_json_path = Path(os.environ.get("QG_JSON_PATH", r"testing\custom_testing_inputs\question_guidelines1.json"))
    config_yaml_path = Path(os.environ.get("CONFIG_YAML_PATH", "config.yaml"))
    output_txt_path = Path(os.environ.get("OUTPUT_TXT_PATH", r"testing\op7.txt"))

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
    graph = AgendaGenerationAgent.get_graph()
    otpt = asyncio.run(graph.ainvoke(inp, config))

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
