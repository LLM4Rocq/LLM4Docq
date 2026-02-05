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

from rocq_ml_toolbox.parser.rocq_parser import RocqParser, Source, VernacElement, VernacKind
from rocq_ml_toolbox.inference.client import PytanqueExtended

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


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
    def __init__(self, client:PytanqueExtended, output_dir: Path):
        self.output_dir = output_dir
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

    def process_file(self, filepath: Path) -> Optional[str]:
        """
        Process a single .v file: extract AST, annotate, chunk, and save.
        Returns the filename on success, None on failure.
        """
        try:
            source = Source.from_local_path(str(filepath))
            target_elements, proof_elements = self.parser.ast(source)
            
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

            rel_name = filepath.stem
            export_folder = self.output_dir / rel_name
            export_folder.mkdir(parents=True, exist_ok=True)

            for k, (chunk_text, elements) in enumerate(chunks):
                (export_folder / f'chunk_doc_{k}.v').write_text(chunk_text, encoding='utf-8')
                with open(export_folder / f'chunk_elements_{k}.json', 'w') as f:
                    json.dump([e.to_json() for e in elements], f, indent=4)

            (export_folder / 'source.v').write_text(src_wo_proof_str, encoding='utf-8')

            return str(filepath)

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Extract and annotate Rocq/Coq vernacular elements.")
    parser.add_argument("input_dir", type=Path, help="Directory containing .v files")
    parser.add_argument("--output-dir", type=Path, default=Path("./export"), help="Output directory")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Pytanque server host")
    parser.add_argument("--port", type=int, default=5000, help="Pytanque server port")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads")
    
    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    files = list(args.input_dir.rglob("*.v"))
    files = [Path(p) for p in files]
    if not files:
        logger.warning("No .v files found in input directory.")
        sys.exit(0)
    
    logger.info(f"Found {len(files)} files to process.")

    client = PytanqueExtended(args.host, args.port)
    client.connect()
    extractor = RocqExtractor(client, args.output_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(extractor.process_file, files), 
            total=len(files), 
            desc="Processing"
        ))

    success_count = sum(1 for r in results if r is not None)
    logger.info(f"Completed. Successfully processed {success_count}/{len(files)} files.")

if __name__ == "__main__":
    main()