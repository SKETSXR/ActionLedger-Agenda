# =============================================================================
# Script: app.py
# =============================================================================
# Purpose
#   Load test inputs (JD text, CV PDF, Skill Tree JSON, Question Guidelines),
#   parse JD/CV via existing tools, build the pipeline InputSchema, execute the
#   AgendaGenerationAgent graph, print artifacts, and write a combined output file.
#
# Behavior
#   - Reads from the same hardcoded paths
#   - Uses the same parse functions and error messages
#   - Prints the same debug dumps before running the graph
#   - Invokes the compiled graph with the loaded config
#   - Prints each output section and writes a combined text file
#   - Reports total time in minutes
# =============================================================================

import asyncio
import json
import time
import yaml

from pathlib import Path

from dotenv import load_dotenv

from src.agent.AgendaGenerationAgent import AgendaGenerationAgent
from src.schema.input_schema import (
    JobDescriptionSchema,
    CandidateProfileSchema,
    SkillTreeSchema,
    InputSchema,
)
from src.tools.jd_extraction import parse_jd_text_to_json
from src.tools.cv_extraction import parse_pdf_to_json


# ----------------------------- Helpers ----------------------------- #

def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _skill_tree_from_dict(tree_dict: dict) -> SkillTreeSchema:
    # Construct SkillTreeSchema from a JSON dict (unchanged behavior)
    return SkillTreeSchema(**tree_dict)


# ----------------------------- Main ----------------------------- #

if __name__ == "__main__":
    start_ns = time.time_ns()
    load_dotenv()

    # Paths (kept identical semantics; normalized through Path)
    jd_txt_path = Path(r"testing\Ayam\jd.txt")
    cv_pdf_path = Path(r"testing\Ayam\AyamHeniberMeitei_2025L - ayam heniber.pdf")
    skill_tree_json_path = Path(r"testing\custom_testing_inputs\skilltree3_priority.json")
    qg_json_path = Path(r"testing\custom_testing_inputs\question_guidelines1.json")
    config_yaml_path = Path("config.yaml")
    output_txt_path = Path(r"testing\op7.txt")

    # --- JD parse ---
    jd_text = _read_text(jd_txt_path)
    jd_json_str = asyncio.run(parse_jd_text_to_json(jd_text))
    if jd_json_str == "JD not contain any text":
        raise RuntimeError("Open AI API not running")

    jd_obj = json.loads(jd_json_str)
    jdes = JobDescriptionSchema(
        job_role=jd_obj["job_role"],
        company_background=jd_obj["company_background"],
        fundamental_knowledge=jd_obj.get("fundamental_knowledge"),
    )

    # --- CV parse ---
    cv_json_str = asyncio.run(parse_pdf_to_json(str(cv_pdf_path)))
    if cv_json_str == "CV does not contain proper text":
        raise RuntimeError("Open AI API not running")
    cv_obj = json.loads(cv_json_str)

    cp = CandidateProfileSchema(
        skills=cv_obj["skills"],
        projects=cv_obj["projects"],
        experience=cv_obj["experience"],
    )

    # --- Skill tree ---
    skill_tree_dict = _read_json(skill_tree_json_path)
    skill_tree = _skill_tree_from_dict(skill_tree_dict)

    # --- Question guidelines ---
    question_guidelines = _read_json(qg_json_path)

    # Build the input schema for the pipeline
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

    # Load config and run the graph
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
