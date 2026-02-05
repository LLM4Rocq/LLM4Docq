#!/usr/bin/env python3
"""
Rocq Prover Annotation Pipeline (Step 2)
Takes extracted chunks from Step 1, generates headers, and uses an LLM to annotate elements.
"""

import argparse
import concurrent.futures
import re
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def clean_json_response(response_text: str) -> str:
    """Removes Markdown code blocks if present to ensure json.loads works."""
    pattern = r"^```(?:json)?\s*(.*)\s*```$"
    match = re.match(pattern, response_text.strip(), re.DOTALL)
    if match:
        return match.group(1)
    return response_text

def load_file_content(path: Path) -> str:
    return path.read_text(encoding='utf-8')


class LLMAnnotator:
    def __init__(self, config_path: Path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.client = OpenAI()
        
        self.model = self.config['model']
        self.params = self.config['parameters']
        self.max_retries = self.config['max_retries']

    def call_llm(self, prompt: str) -> str:
        """Raw call to OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, # Force JSON mode
                **self.params
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Call failed: {e}")
            return "{}"

class AnnotationPipeline:
    def __init__(self, annotator: LLMAnnotator, templates_dir: Path):
        self.annotator = annotator
        self.header_template = load_file_content(templates_dir / 'prompt_header.txt')
        self.task_template = load_file_content(templates_dir / 'prompt_task.txt')

    def generate_header(self, source_path: Path) -> str:
        """Step 1: Generate the context header from source.v"""
        source_content = load_file_content(source_path)
        
        header_prompt = self.header_template.replace('{source}', source_content)
        header_context = self.annotator.call_llm(header_prompt)
        try:
            data = json.loads(header_context)
            if isinstance(data, dict) and 'header' in data:
                return data['header']
            return header_context
        except:
            return header_context

    def process_chunk(self, 
                      chunk_doc_path: Path, 
                      chunk_elements_path: Path, 
                      header_context: str) -> Optional[Dict[str, str]]:
        """
        Step 2: Annotate a specific chunk with retry logic for missing elements.
        """
        try:
            chunk_doc = load_file_content(chunk_doc_path)
            elements_data = json.loads(load_file_content(chunk_elements_path))
            required_uids = {e['data']['uid'] for e in elements_data}
            
            if not required_uids:
                return {}

            final_annotations: Dict[str, str] = {}
            missing_uids = required_uids.copy()
            attempts = 0

            while missing_uids and attempts < self.annotator.max_retries:
                attempts += 1
                
                missing_list_str = "\n".join([f"- {uid}" for uid in missing_uids])
                
                prompt = self.task_template \
                    .replace('{header}', header_context) \
                    .replace('{source}', chunk_doc) \
                    .replace('{missing}', missing_list_str)

                response_text = self.annotator.call_llm(prompt)
                
                try:
                    cleaned_json = clean_json_response(response_text)
                    batch_result = json.loads(cleaned_json)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON for {chunk_doc_path.name} (Attempt {attempts})")
                    continue

                for uid, docstring in batch_result.items():
                    if uid in missing_uids:
                        final_annotations[uid] = docstring
                        missing_uids.remove(uid)
                
                if missing_uids:
                    logger.info(f"Chunk {chunk_doc_path.name}: Missing {len(missing_uids)} elements. Retrying...")

            if missing_uids:
                logger.warning(f"Chunk {chunk_doc_path.name}: Failed to annotate {missing_uids} after retries.")

            return final_annotations

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_doc_path}: {e}")
            return None

    def process_folder(self, folder_path: Path) -> None:
        """
        Orchestrates processing for a single source folder (created in Step 1).
        """
        source_file = folder_path / 'source.v'
        if not source_file.exists():
            return

        header = self.generate_header(source_file)

        chunk_docs = sorted(list(folder_path.glob('chunk_doc_*.v')))
        
        for chunk_doc in chunk_docs:
            match = re.search(r'chunk_doc_(\d+)\.v', chunk_doc.name)
            if not match: continue
            k = match.group(1)
            
            chunk_elem = folder_path / f'chunk_elements_{k}.json'
            if not chunk_elem.exists(): continue

            annotations = self.process_chunk(chunk_doc, chunk_elem, header)
            
            if annotations:
                output_path = folder_path / f'annotations_{k}.json'
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="LLM Annotation Pipeline for Rocq")
    parser.add_argument("input_dir", type=Path, help="Root folder containing exported source folders (from Step 1)")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config")
    parser.add_argument("--templates", type=Path, default=Path("templates"), help="Folder containing prompt_header.txt and prompt_task.txt")
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

    folders = [f for f in args.input_dir.iterdir() if f.is_dir()]
    
    if not folders:
        logger.warning("No folders found in input directory.")
        sys.exit(0)

    logger.info(f"Found {len(folders)} folders to process.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(pipeline.process_folder, folders), 
            total=len(folders), 
            desc="Annotating Sources"
        ))

    logger.info("Annotation pipeline completed.")

if __name__ == "__main__":
    main()