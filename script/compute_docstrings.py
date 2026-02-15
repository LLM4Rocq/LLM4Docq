#!/usr/bin/env python3
"""
Rocq Prover Annotation Pipeline (Step 2)
Takes extracted chunks from Step 1, generates headers, and uses an LLM to annotate elements.
"""

import argparse
import concurrent.futures
import json
import re
import sys
import yaml
import logging
import random

from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

import openai
from openai import OpenAI
from tqdm import tqdm

from .duplicate import find_source_duplicates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_file_content(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def clean_json_response(response_text: str) -> str:
    """Removes Markdown code blocks if present to ensure json.loads works."""
    pattern = r"^```(?:json)?\s*(.*)\s*```$"
    match = re.match(pattern, response_text.strip(), re.DOTALL)
    if match:
        return match.group(1)
    return response_text

def clean_coq_response(response_text: str) -> str:
    """Removes Markdown code blocks if present to ensure json.loads works."""
    pattern = r"^```(?:coq)?\s*(.*)\s*```$"
    match = re.match(pattern, response_text.strip(), re.DOTALL)
    if match:
        return match.group(1)
    return response_text

def docstrings_json_schema(allowed_uids: List[str]) -> Dict[str, Any]:
    # We constrain uid via enum to guarantee no unexpected keys.
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "string", "enum": allowed_uids},
                        "docstring": {"type": "string"}
                    },
                    "required": ["uid", "docstring"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["items"],
        "additionalProperties": False
    }

def items_to_mapping(payload: Dict[str, Any]) -> Dict[str, str]:
    # Converts {"items": [{"uid": "...", "docstring": "..."}, ...]} into {"uid": "docstring", ...}
    items = payload.get("items", [])
    out: Dict[str, str] = {}
    for it in items:
        uid = it.get("uid")
        doc = it.get("docstring")
        if isinstance(uid, str) and isinstance(doc, str):
            out[uid] = doc
    return out


class LLMAnnotator:
    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")
        self.client = OpenAI(base_url=base_url)

        self.params = self.config.get("parameters", {}) or {}
        self.max_retries = int(self.config.get("max_retries", 0))

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        OpenAI-compatible call to SGLang.
        Returns assistant message.content (expected to be JSON text when constrained).
        """

        resp = self.client.chat.completions.create(
            messages=messages,
            response_format=response_format,
            **self.params,
        )

        msg = resp.choices[0].message
        return msg.content or ""


class AnnotationPipeline:
    def __init__(self, annotator: LLMAnnotator, templates_dir: Path):
        self.annotator = annotator
        self.header_template = load_file_content(templates_dir / "prompt_header.txt")
        self.one_liner_template = load_file_content(templates_dir / "prompt_one_liner.txt")
        self.task_template = load_file_content(templates_dir / "prompt_docstrings.txt")

    def generate_one_liner(self, source_path: Path) -> str:
        """Generate the context header from source.v with structured output."""
        source_content = load_file_content(source_path)
        header_prompt = self.one_liner_template.replace("{source}", source_content)

        response_text = self.annotator.call_llm(
            messages=[{"role": "user", "content": header_prompt}]
        )

        # Structured outputs should already be pure JSON, but keep a defensive clean.
        try:
            cleaned = clean_coq_response(response_text)
            return cleaned
        except Exception:
            pass
        
        logger.warning(
            f"One liner {source_path.name}: Failed to extract coq response."
        )
        # Fallback: return raw text if something unexpected happens.
        return response_text
    
    def generate_header(self, source_path: Path) -> str:
        """Generate the context header from source.v with structured output."""
        source_content = load_file_content(source_path)
        header_prompt = self.header_template.replace("{source}", source_content)

        response_text = self.annotator.call_llm(
            messages=[{"role": "user", "content": header_prompt}],
        )

        # Structured outputs should already be pure JSON, but keep a defensive clean.
        try:
            cleaned = clean_coq_response(response_text)
            return cleaned
        except Exception:
            pass
        
        logger.warning(
            f"Header {source_path.name}: Failed to extract coq response."
        )
        # Fallback: return raw text if something unexpected happens.
        return response_text

    def process_chunk(
        self,
        chunk_doc_path: Path,
        chunk_elements_path: Path,
        header_context: str,
        done_uid: Set[str]=set(),
        cache: Dict[str, str]={}
    ) -> Optional[Dict[str, str]]:
        """
        Annotate a specific chunk with retry logic for missing elements.
        Returns {uid: docstring}.
        """
        try:
            chunk_doc = load_file_content(chunk_doc_path)
            elements_data = json.loads(load_file_content(chunk_elements_path))
            missing_uids:List[str] = [e["data"]["uid"] for e in elements_data if e["data"]["uid"] not in done_uid]
            final_annotations: Dict[str, str] = {}

            new_missing_uids = []
            for uid in missing_uids:
                if uid in cache:
                    final_annotations[uid] = cache[uid]
                elif not uid.startswith('None') and not uid.startswith('NOTATION'):
                    # required unambiguous identifier
                    new_missing_uids.append(uid)

            if not missing_uids:
                return final_annotations

            attempts = 0

            while missing_uids and attempts < self.annotator.max_retries + 1:
                attempts += 1

                # Constrain output uid to exactly the missing set for this attempt.
                allowed_uids = missing_uids

                missing_list_str = "\n".join([f"- {uid}" for uid in allowed_uids])
                prompt = (
                    self.task_template
                    .replace("{header}", header_context)
                    .replace("{source}", chunk_doc)
                    .replace("{missing}", missing_list_str)
                )

                response_text = self.annotator.call_llm(
                    messages=[{"role": "user", "content": prompt}],
                    # response_format={
                    #     "type": "json_schema",
                    #     "json_schema": {
                    #         "name": "docstring_map",
                    #         "schema": docstrings_json_schema(allowed_uids),
                    #         "strict": True,
                    #     },
                    # }
                )

                try:
                    cleaned = clean_json_response(response_text)
                    batch_map = json.loads(cleaned)
                    # batch_map = items_to_mapping(payload)
                except Exception:
                    logger.warning(
                        f"Failed to parse structured JSON for {chunk_doc_path.name} (Attempt {attempts})"
                    )
                    continue

                # Accept only uids that were missing this round.
                for uid, docstring in batch_map.items():
                    if uid in missing_uids:
                        final_annotations[uid] = docstring
                        missing_uids.remove(uid)

                if missing_uids:
                    logger.info(
                        f"Chunk {chunk_doc_path.name}: Missing {len(missing_uids)} elements. Retrying..."
                    )

            if missing_uids:
                logger.warning(
                    f"Chunk {chunk_doc_path.name}: Failed to annotate {sorted(missing_uids)} after retries."
                )

            return final_annotations

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_doc_path}: {e}")
            return None

    def process_folder(self, folder_path: Path, folder_copy: List[Path]=[]) -> None:
        """Orchestrates processing for a single source folder (created in Step 1)."""
        source_file = folder_path / "source.v"
        if not source_file.exists():
            return

        lock_path = folder_path / '_lock'
        if lock_path.exists():
            return

        lock_paths = [lock_path] + [path_copy / '_lock' for path_copy in folder_copy]
        for lock_path in lock_paths:
            with open(lock_path, 'w') as file:
                pass
        try:        
            one_liner_path = folder_path / 'one_liner.txt'
            if not one_liner_path.exists():
                one_liner = self.generate_one_liner(source_file)
                with open(one_liner_path, 'w') as file:
                    file.write(one_liner)
                for path_copy in folder_copy:
                    one_liner_copy_path = path_copy / "one_liner.txt"
                    with open(one_liner_copy_path, "w") as file:
                        file.write(one_liner)

        
            header_path = folder_path / 'header.txt'
            if not header_path.exists():
                header = self.generate_header(source_file)
                with open(header_path, 'w') as file:
                    file.write(header)
                for path_copy in folder_copy:
                    header_copy_path = path_copy / "header.txt"
                    with open(header_copy_path, "w") as file:
                        file.write(header)
            
            with open(header_path, 'r') as file:
                header = file.read()
            chunk_docs = sorted(list(folder_path.glob("chunk_doc_*.v")))

            cache_elem_path = folder_path / f"cache.json"
            cache_elem = {}
            if cache_elem_path.exists():
                with open(cache_elem_path, 'r') as file:
                    cache_elem = json.load(file)
            
            for chunk_doc in chunk_docs:
                match = re.search(r"chunk_doc_(\d+)\.v", chunk_doc.name)
                if not match:
                    continue
                k = match.group(1)

                chunk_elem = folder_path / f"chunk_elements_{k}.json"
                if not chunk_elem.exists():
                    continue
                
                output_path = folder_path / f"annotations_{k}.json"

                if not output_path.exists():
                    new_annotations = self.process_chunk(chunk_doc, chunk_elem, header, cache=cache_elem)
                else:
                    with open(output_path, 'r') as file:
                        prev_annotations = json.load(file)
                    new_annotations = self.process_chunk(chunk_doc, chunk_elem, header, done_uid=set(prev_annotations.keys()), cache=cache_elem)
                    if new_annotations:
                        new_annotations = prev_annotations | new_annotations
                
                if new_annotations:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(new_annotations, f, indent=4, ensure_ascii=False)
                    
                    for path_copy in folder_copy:
                        output_copy_path = path_copy / f"annotations_{k}.json"
                        with open(output_copy_path, "w", encoding="utf-8") as f:
                            json.dump(new_annotations, f, indent=4, ensure_ascii=False)
        except openai.BadRequestError:
            # IGNORE BadRequest, 
            pass
        finally:
            for lock_path in lock_paths:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

def main():
    parser = argparse.ArgumentParser(description="LLM Annotation Pipeline for Rocq")
    parser.add_argument("--input-dir", type=Path, default=Path("export/"), help="Root folder containing exported source folders (from Step 1)")
    parser.add_argument("--config", type=Path, default=Path("config/annotator/config.yaml"), help="Path to YAML config")
    parser.add_argument("--templates", type=Path, default=Path("config/annotator"), help="Folder containing prompt_header.txt and prompt_docstrings.txt")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel folder processors")

    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error("Input directory not found.")
        sys.exit(1)
    try:
        annotator = LLMAnnotator(args.config)
        pipeline = AnnotationPipeline(annotator, args.templates)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    mapping_to_do = {}
    for list_path in find_source_duplicates(args.input_dir):
        list_path_parent = [Path(p).parent for p in list_path]
        mapping_to_do[list_path_parent[0]] = list_path_parent[1:]
    
    items = list(mapping_to_do.items())
    random.shuffle(items)
    folders, copy_folders = zip(*items)
    if not folders:
        logger.warning("No folders found in input directory.")
        sys.exit(0)
    logger.info(f"Found {len(folders)} folders to process.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(
            executor.map(pipeline.process_folder, folders, copy_folders),
            total=len(folders),
            desc="Annotating Sources"
        ))

    logger.info("Annotation pipeline completed.")


if __name__ == "__main__":
    main()
