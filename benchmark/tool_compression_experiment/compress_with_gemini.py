#!/usr/bin/env python3
"""Compress MCP tool descriptions with Gemini, preserving signatures."""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from workflow.executor import Executor

OUTPUT = Path(__file__).with_name("gemini_compressed_descriptions.json")
MODEL = "gemini/gemini-2.5-flash"


async def main():
    executor = Executor(llm=None)
    original = await executor.get_agent_descriptions()

    if not original:
        raise RuntimeError("No MCP descriptions discovered.")

    checkpoint = OUTPUT.with_name("gemini_compression_checkpoint.json")
    completed = {}

    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("model") != MODEL or saved.get("original_descriptions") != original:
            raise RuntimeError("Checkpoint does not match current model or tool descriptions.")
        completed = saved["completed"]
        print(f"Resuming from {len(completed)} completed tools.", flush=True)

    compressed = {}
    counts = {"original_chars": 0, "compressed_chars": 0, "tools": 0}

    for agent, description_block in original.items():
        if "(unavailable:" in description_block:
            raise RuntimeError(f"{agent} unavailable; refusing partial output.")

        new_lines = []

        tool_blocks = re.split(
            r"(?=^  - [^\n]+\([^\n]*\): )",
            description_block,
            flags=re.MULTILINE,
        )

        for block in tool_blocks:
            if not block.strip():
                continue
            match = re.match(
                r"^(\s+-\s+.+?\([^)]*\):\s*)([\s\S]*)$",
                block,
            )
            if not match:
                raise ValueError(f"Unexpected tool block: {block[:120]!r}")

            signature, description = match.groups()
            description = description.strip()

            tool_key = f"{agent}::{signature.strip()}"
            if tool_key in completed:
                shortened = completed[tool_key]
                print(f"Reusing: {tool_key}", flush=True)
            else:
                response = litellm.completion(
                    model=MODEL,
                    api_key=os.environ["GEMINI_API_KEY"],
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress MCP tool descriptions for a planning agent. "
                                "Return only a concise replacement description. "
                                "Preserve all behavioral details, constraints, "
                                "parameter meanings, and important distinctions. "
                                "Do not invent capabilities."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Tool: {signature.strip()}\nOriginal description: {description}",
                        },
                    ],
                    max_tokens=4000,
                    temperature=0,
                )

                finish_reason = response.choices[0].finish_reason
                if finish_reason != "stop":
                    raise RuntimeError(
                        f"Gemini did not finish normally for {agent}: "
                        f"{signature.strip()} (finish_reason={finish_reason})"
                    )

                shortened = (response.choices[0].message.content or "").strip()
                if not shortened:
                    raise RuntimeError(f"Empty Gemini response for {signature}")

                shortened = " ".join(shortened.split())

                completed[tool_key] = shortened
                checkpoint.write_text(json.dumps({
                    "model": MODEL,
                    "original_descriptions": original,
                    "completed": completed,
                }, indent=2, ensure_ascii=False))

            new_lines.append(signature + shortened)
            counts["original_chars"] += len(description)
            counts["compressed_chars"] += len(shortened)
            counts["tools"] += 1

            print(f"{agent}: {signature.strip()}", flush=True)

        compressed[agent] = "\n".join(new_lines)

    payload = {
        "model": MODEL,
        "original_descriptions": original,
        "compressed_descriptions": compressed,
        "statistics": counts,
    }

    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUTPUT}")
    print(f"Tools: {counts['tools']}")
    print(f"Original description characters: {counts['original_chars']}")
    print(f"Compressed description characters: {counts['compressed_chars']}")


if __name__ == "__main__":
    asyncio.run(main())
