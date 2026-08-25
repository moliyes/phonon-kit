from __future__ import annotations

import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PROJECT_ROOT / "skills" / "materials-science" / "phonon-kit"
SKILL_PATH = SKILL_ROOT / "SKILL.md"
EXPECTED_REFERENCES = {
    "references/configuration.md",
    "references/qha-workflow.md",
    "references/anharmonic-workflow.md",
    "references/results-and-recovery.md",
    "references/scientific-interpretation.md",
}


def _load_skill() -> tuple[dict, str]:
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    _, frontmatter, body = content.split("---", 2)
    return yaml.safe_load(frontmatter), body.strip()


def test_hermes_frontmatter_and_trigger_description() -> None:
    frontmatter, body = _load_skill()

    assert frontmatter["name"] == "phonon-kit"
    assert frontmatter["version"] == "0.2.0"
    assert frontmatter["platforms"] == ["linux"]
    assert frontmatter["author"] == "moliyes, Hermes Agent"
    assert body

    description = frontmatter["description"]
    assert len(description) <= 60
    assert description.endswith(".")
    for term in ("Phonopy", "Phono3py", "DeepMD", "VASP", "QHA", "phase"):
        assert term.lower() in description.lower()

    hermes = frontmatter["metadata"]["hermes"]
    assert hermes["category"] == "materials-science"
    assert hermes["requires_tools"] == ["terminal"]
    assert hermes["related_skills"] == ["calypso-dp-search"]
    assert [item["key"] for item in hermes["config"]] == [
        "phonon_kit.project_root"
    ]


def test_references_are_direct_complete_and_portable() -> None:
    _, body = _load_skill()
    references = set(re.findall(r"references/[a-z0-9-]+\.md", body))
    assert references == EXPECTED_REFERENCES

    skill_files = [SKILL_PATH]
    for relative in references:
        assert relative.count("/") == 1
        path = SKILL_ROOT / relative
        assert path.is_file()
        skill_files.append(path)

    for path in skill_files:
        content = path.read_text(encoding="utf-8")
        assert "/home/begin" not in content
        if path != SKILL_PATH and len(content.splitlines()) > 100:
            assert "## Contents" in content


def test_skill_only_documents_real_ph_commands() -> None:
    content = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [SKILL_PATH, *(SKILL_ROOT / ref for ref in EXPECTED_REFERENCES)]
    )
    top_level = set(re.findall(r"\bph\s+(--version|[a-z][a-z-]*)", content))
    qha_subcommands = set(re.findall(r"\bph\s+qha\s+([a-z][a-z-]*)", content))
    anh_subcommands = set(re.findall(r"\bph\s+anh\s+([a-z][a-z-]*)", content))

    assert top_level <= {
        "--version",
        "init",
        "validate",
        "run",
        "resume",
        "collect",
        "status",
        "plot",
        "qha",
        "anh",
    }
    assert qha_subcommands <= {
        "init",
        "validate",
        "plan",
        "run",
        "resume",
        "collect",
        "status",
        "plot",
    }
    assert {"init", "validate", "run", "resume", "collect", "status", "plot"} <= top_level
    assert {"init", "validate", "plan", "run", "resume", "collect", "status", "plot"} <= qha_subcommands
    assert anh_subcommands <= {"init", "plan", "run", "status", "plot"}
    assert {"init", "plan", "run", "status", "plot"} <= anh_subcommands


def test_generated_agent_metadata_matches_skill() -> None:
    metadata = yaml.safe_load(
        (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    )
    interface = metadata["interface"]
    assert interface["display_name"] == "Phonon Kit"
    assert 25 <= len(interface["short_description"]) <= 64
    assert "$phonon-kit" in interface["default_prompt"]
