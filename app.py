import asyncio
import json
import time
from pathlib import Path

import yaml
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
def _load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_skill_tree(tree_json: dict) -> SkillTreeSchema:
    # Construct SkillTreeSchema from a JSON dict (unchanged behavior)
    return SkillTreeSchema(**tree_json)


# ----------------------------- Main ----------------------------- #
if __name__ == "__main__":
    start = time.time_ns()
    load_dotenv()

    # Paths (kept same semantics; normalized through Path)
    jd_txt_path = Path(r"testing\Ayam\jd.txt")
    cv_pdf_path = Path(r"testing\Ayam\AyamHeniberMeitei_2025L - ayam heniber.pdf")
    skill_tree_json_path = Path(r"testing\custom_testing_inputs\skilltree3_priority.json")
    qg_json_path = Path(r"testing\custom_testing_inputs\question_guidelines1.json")
    config_yaml_path = Path("config.yaml")
    output_txt_path = Path(r"testing\op1.txt")

    # --- JD parse ---
    jd_inp_text = _load_text(jd_txt_path)
    jd_json_string = asyncio.run(parse_jd_text_to_json(jd_inp_text))
    if jd_json_string == "JD not contain any text":
        raise RuntimeError("Open AI API not running")

    jd = json.loads(jd_json_string)
    jdes = JobDescriptionSchema(
        job_role=jd["job_role"],
        company_background=jd["company_background"],
        fundamental_knowledge=jd.get("fundamental_knowledge"),
    )

    # --- CV parse ---
    candidate_profile = asyncio.run(parse_pdf_to_json(str(cv_pdf_path)))
    if candidate_profile == "CV does not contain proper text":
        raise RuntimeError("Open AI API not running")
    candidate_profile = json.loads(candidate_profile)

    cp = CandidateProfileSchema(
        skills=candidate_profile["skills"],
        projects=candidate_profile["projects"],
        experience=candidate_profile["experience"],
    )

    # --- Skill tree ---
    tree_data = _load_json(skill_tree_json_path)
    root = _load_skill_tree(tree_data)

    # --- Question guidelines ---
    question_guidelines = _load_json(qg_json_path)

    # Build the input schema for the pipeline
    inp = InputSchema(
        job_description=jdes,
        skill_tree=root,
        candidate_profile=cp,
        question_guidelines=question_guidelines,
    )

    # Debug prints (preserved)
    print(inp.candidate_profile.model_dump_json(indent=2) + "\n")
    print(inp.job_description.model_dump_json(indent=2) + "\n")
    print(inp.skill_tree.model_dump_json(indent=2))

    # Load config and run the graph
    config = _load_yaml(config_yaml_path)
    graph = AgendaGenerationAgent.get_graph()
    otpt = asyncio.run(graph.ainvoke(inp, config))

    # Collect & save outputs
    combined_text = ""
    for k, v in otpt.items():
        k_cap = str(k).capitalize()
        body = v.model_dump_json(indent=2)
        print(f"\n{k_cap} --->\n\n {body}\n")
        combined_text += f"\n{k_cap} --->\n\n {body}\n"

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_txt_path.open("w", encoding="utf-8") as f:
        f.write(combined_text)

    end = time.time_ns()
    print(f"\nTime taken: {(end - start) / 60000000000} mins")
