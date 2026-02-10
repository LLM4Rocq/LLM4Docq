#!/usr/bin/env python3
"""
Rocq Prover Extraction Pipeline
Extracts, annotates, and chunks vernacular elements from Rocq (.v) files for LLM processing.
"""

import argparse
import concurrent.futures
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from tqdm import tqdm

from rocq_ml_toolbox.docker.opam_docker import OpamDocker, OpamConfig
from rocq_ml_toolbox.parser.rocq_parser import RocqParser, Source, VernacElement, VernacKind
from rocq_ml_toolbox.inference.client import PytanqueExtended

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def split_at_dir(p: Path, marker: str) -> Tuple[Path, Path]:
    try:
        parts = p.parts
        i = parts.index(marker)
        # edge case is already well managed by Path (i.e. if marker is last available dir)
        return Path(*parts[:i+1]), Path(*parts[i+1:])
    except ValueError:
        raise ValueError(f"{marker!r} not found in path")

def remove_ocaml_comments(src: str, strict: bool = False) -> str:
    """
    Remove OCaml-style comments (* ... *) from source string.
    Handles nested comments and string/char literals.
    """
    lines: List[Tuple[str, bool]] = []
    buf: List[str] = []
    had_comment = False
    i, n = 0, len(src)
    depth = 0
    in_string = False
    in_char = False
    escape = False

    def flush_line(eol: str = "") -> None:
        nonlocal buf, had_comment
        lines.append(("".join(buf), had_comment))
        buf = []
        had_comment = False

    while i < n:
        # Handle EOL
        if src[i] == "\n":
            if depth > 0: had_comment = True
            else: buf.append("\n")
            flush_line()
            i += 1; continue
        if src[i] == "\r":
            eol = "\r\n" if i + 1 < n and src[i + 1] == "\n" else "\r"
            if depth > 0: had_comment = True
            else: buf.append(eol)
            flush_line()
            i += len(eol); continue

        # Handle Comments
        if depth > 0:
            if i + 1 < n and src[i:i+2] == "(*":
                depth += 1; i += 2
            elif i + 1 < n and src[i:i+2] == "*)":
                depth -= 1; i += 2
            else:
                i += 1
            had_comment = True
            continue

        ch = src[i]
        
        # Handle String/Char Literals
        if in_string:
            buf.append(ch)
            if escape: escape = False
            elif ch == "\\": escape = True
            elif ch == '"': in_string = False
            i += 1; continue

        if in_char:
            buf.append(ch)
            if escape: escape = False
            elif ch == "\\": escape = True
            elif ch == "'": in_char = False
            i += 1; continue

        # Start Comment or Literal
        if i + 1 < n and src[i:i+2] == "(*":
            depth = 1; had_comment = True; i += 2
        elif ch == '"':
            in_string = True; buf.append(ch); i += 1
        elif ch == "'":
            in_char = True; buf.append(ch); i += 1
        else:
            buf.append(ch); i += 1

    if strict and depth != 0:
        raise ValueError("Unclosed OCaml comment")
    
    if buf:
        lines.append(("".join(buf), had_comment))

    return "".join(
        text for (text, touched) in lines
        if not (touched and text.strip() == "")
    )

def normalize_text(text: str) -> str:
    """Normalizes whitespace and reduces consecutive newlines."""
    lines = [line.strip() for line in text.splitlines()]
    new_text = '\n'.join(lines)
    while '\n\n\n' in new_text:
        new_text = new_text.replace('\n\n\n', '\n\n')
    return new_text


class RocqExtractor:
    def __init__(self, client:PytanqueExtended):
        self.parser = RocqParser(client)

    def chunk_doc(self, content: str, uid_map: Dict[str, VernacElement], 
                 prefix_size: int = 50, max_lines: int = 100, 
                 max_elements: int = 25) -> List[Tuple[str, List[VernacElement]]]:
        
        pattern = re.compile(r'<annotation>(.*)</annotation>')
        chunks = []
        
        curr_prefix: List[str] = []
        curr_lines: List[str] = []
        curr_elements: List[VernacElement] = []
        is_close = True

        for line in content.splitlines():
            m = pattern.match(line)

            # Check if we should flush the current chunk
            should_flush = is_close and (len(curr_lines) >= max_lines or len(curr_elements) > max_elements)
            
            if should_flush:
                full_doc = '\n'.join(curr_lines)
                context_prefix = '\n'.join(curr_prefix[-prefix_size:])
                chunks.append((context_prefix + full_doc, curr_elements))

                curr_prefix.extend([l for l in curr_lines if not l.startswith('<annotation>')])
                curr_lines = []
                curr_elements = []

            if m:
                uid = m.group(1)
                if uid in uid_map:
                    curr_elements.append(uid_map[uid])
                curr_lines.append(line)
                is_close = False
            elif line == '<END>':
                is_close = True
            else:
                curr_lines.append(line)

        if curr_elements:
            full_doc = '\n'.join(curr_lines)
            context_prefix = '\n'.join(curr_prefix[-prefix_size:])
            chunks.append((context_prefix + full_doc, curr_elements))
            
        return chunks

    def process_file(self, source: Source, output_dir: Path, target_name: Optional[str]=None, force_process=False) -> bool:
        """
        Process a single .v file: extract AST, annotate, chunk, and save.
        Returns the filename on success, None on failure.
        """
        filepath = Path(source.path)
        root = None
        tail = Path("")
        rel_name = filepath.stem
        if target_name:
            root, tail = split_at_dir(Path(source.path), target_name)
            tail = tail.parent
            export_folder = output_dir / target_name / tail / rel_name
        else:
            export_folder = output_dir / rel_name
        export_folder.mkdir(parents=True, exist_ok=True)
        output_final = (export_folder / 'source.v')
        elements_path = export_folder / 'elements.json'
        if output_final.exists() and not force_process:
            return True
        
        if elements_path.exists() and not force_process:
            with open(elements_path, 'r') as file:
                content = json.load(file)
            target_elements, proof_elements = content['target_elements'], content['proof_elements']
        else:
            target_elements, proof_elements = self.parser.ast(source, root=root)
            with open(elements_path, 'w') as file:
                json.dump({
                    "target": [t.to_json() for t in target_elements],
                    "proof": [t.to_json() for t in proof_elements]
                }, file, indent=4)

        to_do: List[Tuple[int, str, VernacElement]] = []
        idx_notation = 0
        
        for element in target_elements:
            to_do.append((element.span.bp, "COMMENT", element))
            if element.kind in (VernacKind.SYNTACTIC_DEFINITION, VernacKind.NOTATION):
                idx_notation += 1
        
        for element in proof_elements:
            to_do.append((element.span.bp, "DELETE", element))

        to_do.sort(key=lambda x: x[0], reverse=True)

        source_annotated_bytes = source.content_utf8
        source_wo_proof_bytes = source.content_utf8
        uid_vernac_element: Dict[str, VernacElement] = {}
        idx_uid = defaultdict(int)

        for _, task, element in to_do:
            span = element.span
            
            if task == 'COMMENT':
                fqn = element.data.get('fqn', 'unknown')
                if element.kind in (VernacKind.SYNTACTIC_DEFINITION, VernacKind.NOTATION):
                    uid = f'NOTATION_{idx_notation}'
                    idx_notation -= 1
                elif fqn in idx_uid:
                    uid = f"{fqn}_duplicata_{idx_uid[fqn]}"
                else:
                    uid = fqn
                
                idx_uid[fqn] += 1
                element.data['uid'] = uid
                uid_vernac_element[uid] = element

                annotation = f"\n<annotation>{uid}</annotation>\n".encode('utf-8')
                source_annotated_bytes = (
                    source_annotated_bytes[:span.bp] + 
                    annotation + 
                    source_annotated_bytes[span.bp:span.ep] + 
                    b'\n<END>\n' + 
                    source_annotated_bytes[span.ep:]
                )
            
            elif task == 'DELETE':
                source_annotated_bytes = source_annotated_bytes[:span.bp] + source_annotated_bytes[span.ep:]
                source_wo_proof_bytes = source_wo_proof_bytes[:span.bp] + source_wo_proof_bytes[span.ep:]
        src_annotated_str = remove_ocaml_comments(source_annotated_bytes.decode('utf-8'))
        src_wo_proof_str = source_wo_proof_bytes.decode('utf-8')

        src_annotated_str = normalize_text(src_annotated_str)
        src_wo_proof_str = normalize_text(src_wo_proof_str)

        chunks = self.chunk_doc(src_annotated_str, uid_vernac_element)

        for k, (chunk_text, elements) in enumerate(chunks):
            (export_folder / f'chunk_doc_{k}.v').write_text(chunk_text, encoding='utf-8')
            with open(export_folder / f'chunk_elements_{k}.json', 'w') as f:
                json.dump([e.to_json() for e in elements], f, indent=4)
        (export_folder / 'source_annotated.v').write_text(src_annotated_str, encoding='utf-8')
        (export_folder / 'source_wo_proof.v').write_text(src_wo_proof_str, encoding='utf-8')
        (export_folder / 'source.v').write_text(source.content, encoding='utf-8')

        return True

def main():
    parser = argparse.ArgumentParser(description="Extract and annotate Rocq/Coq vernacular elements.")
    parser.add_argument("--config-dir", type=Path, default=Path('Pile-of-rocq/config/'), help="Directory containing yaml config files for Opam Docker")
    parser.add_argument("--output-dir", type=Path, default=Path("./export"), help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")
    
    args = parser.parse_args()
    if not args.config_dir.exists():
        logger.error(f"Input directory does not exist: {args.config_dir}")
        sys.exit(1)

    config_files = list(args.config_dir.rglob("*.yaml"))
    config_files = [Path(p) for p in config_files]
    if not config_files:
        logger.warning("No .yaml files found in input directory.")
        sys.exit(0)
    
    logger.info(f"Found {len(config_files)} config files to process.")

    for config_path in tqdm(config_files):
        # config_path = "Pile-of-rocq/config/coq-mathcomp.yaml"
        print(config_path)
        docker = None
        try:
            config = OpamConfig.from_yaml(config_path)
            output_dir = Path(args.output_dir / config.name)
            docker = OpamDocker(config)
            docker.start_inference_server()

            client = PytanqueExtended("127.0.0.1", 5000)
            client.connect()
            
            for target in config.targets:
                package = target.packages[0]
                target_lib = target.lib
                docker.copy_coq_files_from_package_to_target(package, target_lib)

                output_target_path = output_dir / target_lib 
                output_target_path.mkdir(parents=True, exist_ok=True)
                with open(output_target_path / 'config.json', 'w') as file:
                    json.dump(target.to_json(), file, indent=4)
            list_output_dir = []
            list_targets = []
            list_sources = []
            print("READY")
            
            for target in config.targets:
                target = target.lib
                sub_sources = docker.extract_source_files_from_target(target)
                list_sources.extend(sub_sources)
                list_output_dir.extend([output_dir for _ in sub_sources])
                list_targets.extend([target for _ in sub_sources])
            
            sub_sources = docker.extract_source_files_from_corelib()
            list_sources.extend(sub_sources)
            list_output_dir.extend([output_dir / 'Corelib' for _ in sub_sources])
            with open(output_dir / 'Corelib' / 'config.json', 'w') as file:
                json.dump({
                    "lib": "Corelib",
                    "packages": [
                        f"{config.user}-core.{config.version}"
                    ]
                }, file, indent=4)
            # continue
            list_targets.extend([None for _ in sub_sources])
            extractor = RocqExtractor(client)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = list(tqdm(
                    executor.map(extractor.process_file, list_sources, list_output_dir, list_targets), 
                    total=len(list_sources), 
                    desc="Processing"
                ))

            success_count = sum(1 for r in results if r is not None)
            logger.info(f"Completed {config.name}. Successfully processed {success_count}/{len(results)} files.")
        finally:
            if docker: docker.close()
            pass

if __name__ == "__main__":
    main()