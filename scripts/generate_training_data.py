"""Generates data/training_data.jsonl - the instruction-tuning dataset used
by scripts/train_lora.py to fine-tune the domain LLM.

Fully offline / stdlib-only: it programmatically builds varied
(user parameters) -> (grounded personalized rationale) examples from
data/projects.json, so the base model learns the house style and to only
reference real components for a real project, instead of hallucinating.

Usage:
    python scripts/generate_training_data.py [--out data/training_data.jsonl] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are an assistant that explains why one specific pre-selected "
    "industrial automation / mechatronics project fits a student. "
    "Only mention hardware from the 'have' or 'missing' lists given below - "
    "never invent components that aren't listed. "
    "Reply with ONLY a JSON object: {\"rationale\": \"...\", \"buy_next\": \"...\"}. "
    "'rationale' is 2-3 sentences on why this project fits the student's skill level "
    "and interests. 'buy_next' is one short sentence naming the single most useful "
    "missing part to buy next, or \"nothing - you already have what you need\" if the "
    "missing list is empty."
)

INTEREST_PHRASES = {
    "Sensing & Monitoring": ["monitoring conditions in real time", "sensor data logging", "environment sensing"],
    "Robotics & Manipulation": ["building small robots", "robotic manipulation", "mobile robotics"],
    "Motion Control": ["precise motor control", "CNC-style motion", "positioning systems"],
    "Process Automation & Control": ["automating a process", "closed-loop control", "industrial dosing/batching"],
    "IoT / Industry 4.0": ["connected/IoT devices", "remote telemetry", "predictive maintenance"],
    "Safety & Protection Systems": ["safety interlocks", "alarm systems", "protective automation"],
    "Power & Energy Monitoring": ["energy monitoring", "power/current sensing", "efficiency tracking"],
}

RATIONALE_TEMPLATES = [
    "This {skill} {category_lc} build is a solid fit because you already have {have_list}, "
    "which covers the core of the project, and it lines up well with your interest in {interest}.",
    "Given your {skill} level and interest in {interest}, this project is well scoped: you can "
    "start right away with {have_list} that you already own.",
    "Because you're at {a_skill} stage and care about {interest}, this project makes sense - "
    "it puts {have_list} to direct use and stays within a realistic build time.",
]

BUY_NEXT_TEMPLATES_MISSING = [
    "The most useful part to add next is the {part}.",
    "Pick up a {part} next - it unlocks the rest of the build.",
    "Grab a {part} to complete the required parts list.",
]

BUY_NEXT_NONE = "nothing - you already have what you need"


def human_join(items: list[str]) -> str:
    if not items:
        return "no listed parts yet"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def load_component_names() -> dict:
    raw = json.loads((ROOT / "data" / "components.json").read_text())
    names = {}
    for group in raw.values():
        for c in group:
            names[c["id"]] = c["name"]
    return names


def build_examples(projects: list[dict], seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    names = load_component_names()
    examples = []

    for project in projects:
        required = project["required_components"]
        optional = project["optional_components"]
        interest_options = INTEREST_PHRASES.get(project["category"], [project["category"].lower()])

        # Scenario A: student has everything required.
        # Scenario B: student is missing exactly one required part.
        # Scenario C: student is missing most required parts (only has one, or none).
        scenarios = []
        if required:
            scenarios.append((list(required), []))  # has all
            if len(required) > 1:
                scenarios.append((required[:-1], [required[-1]]))  # missing one
            scenarios.append(([required[0]] if required else [], required[1:]))  # missing most
        else:
            scenarios.append(([], []))

        for have_req, missing_req in scenarios:
            have_opt = optional[: rng.randint(0, len(optional))] if optional else []
            interest = rng.choice(interest_options)

            have_all = [names.get(c, c) for c in (have_req + have_opt)]
            missing_names = [names.get(c, c) for c in missing_req]
            template = rng.choice(RATIONALE_TEMPLATES)
            skill = project["skill_level"]
            a_skill = ("an " if skill[0] in "aeiou" else "a ") + skill
            rationale = template.format(
                skill=skill,
                a_skill=a_skill,
                category_lc=project["category"].lower(),
                have_list=human_join(have_all) if have_all else "the basics you mentioned",
                interest=interest,
            )

            if missing_names:
                buy_next = rng.choice(BUY_NEXT_TEMPLATES_MISSING).format(part=missing_names[0])
            else:
                buy_next = BUY_NEXT_NONE

            user_msg = (
                f"Student skill level: {project['skill_level']}\n"
                f"Student interests: {interest}\n"
                f"Student time budget: {project['time_hours']} hours\n"
                f"Selected project: {project['title']} ({project['category']}, "
                f"{project['skill_level']} level, ~{project['time_hours']}h)\n"
                f"Project description: {project['description']}\n"
                f"Hardware student already has for this project: {have_all or 'none'}\n"
                f"Hardware student is missing for this project: {missing_names or 'none'}\n"
            )
            assistant_msg = json.dumps({"rationale": rationale, "buy_next": buy_next})

            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ]
                }
            )

    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "training_data.jsonl"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    projects = json.loads((ROOT / "data" / "projects.json").read_text())
    examples = build_examples(projects, seed=args.seed)

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} training examples to {out_path}")


if __name__ == "__main__":
    main()
