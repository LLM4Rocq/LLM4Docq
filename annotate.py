from tqdm import tqdm
import time 
import json
import os
import concurrent.futures
from typing import List, Tuple, Dict

from rocq_ml_toolbox.docker.opam_docker import OpamDocker, OpamConfig
from rocq_ml_toolbox.parser.rocq_parser import RocqParser, Source, VernacElement, VernacKind
from rocq_ml_toolbox.inference.client import PytanqueExtended

opam_docker = None

def remove_ocaml_comments(src: str, *, strict: bool = False) -> str:
    """
    Remove OCaml-style comments from `src`.

    - Comments start with "(*" and end with "*)"
    - Can be nested and span multiple lines
    - Delimiters inside string/char literals are ignored
    - Empty/whitespace-only lines are removed ONLY if that line originally contained comment text

    If `strict` is True, raises ValueError on an unclosed comment.
    """
    lines: list[tuple[str, bool]] = []  # (line_text_with_eol, had_comment)
    buf: list[str] = []
    had_comment = False

    i, n = 0, len(src)
    depth = 0
    in_string = False
    in_char = False
    escape = False

    def flush_line(eol: str = "") -> None:
        nonlocal buf, had_comment
        buf.append(eol)
        lines.append(("".join(buf), had_comment))
        buf = []
        had_comment = False

    while i < n:
        # Normalize end-of-line handling while preserving original EOLs.
        if src[i] == "\n":
            if depth > 0:
                had_comment = True
            else:
                buf.append("\n")
            flush_line()
            i += 1
            continue
        if src[i] == "\r":
            eol = "\r\n" if i + 1 < n and src[i + 1] == "\n" else "\r"
            if depth > 0:
                had_comment = True
            else:
                buf.append(eol)
            flush_line()
            i += len(eol)
            continue

        if depth > 0:
            if i + 1 < n and src[i] == "(" and src[i + 1] == "*":
                depth += 1
                had_comment = True
                i += 2
            elif i + 1 < n and src[i] == "*" and src[i + 1] == ")":
                depth -= 1
                had_comment = True
                i += 2
            else:
                had_comment = True
                i += 1
            continue

        ch = src[i]

        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if in_char:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_char = False
            i += 1
            continue

        if i + 1 < n and ch == "(" and src[i + 1] == "*":
            depth = 1
            had_comment = True
            i += 2
        elif ch == '"':
            in_string = True
            buf.append(ch)
            i += 1
        elif ch == "'":
            in_char = True
            buf.append(ch)
            i += 1
        else:
            buf.append(ch)
            i += 1

    if strict and depth != 0:
        raise ValueError("Unclosed OCaml comment")

    if buf:
        lines.append(("".join(buf), had_comment))

    return "".join(
        text for (text, touched) in lines
        if not (touched and text.strip() == "")
    )

def chunk(content: str, max_lines: int, max_element: int, fqn_vernac_element: dict[str, VernacElement]):
    for line in content.splitlines():
        pass

def do_task(source: Source, url: str='127.0.0.1', port: int=5000, attempt:int=3):
    client = PytanqueExtended(url, port)
    client.connect()
    parser = RocqParser(client)

    source_path_normalize = source.path.replace('/', '.')
    export_path = f"export/{source_path_normalize}.jsonl"
    cache = set()
    if os.path.exists(export_path):
        with open(export_path, 'r') as file:
            for line in file:
                content = json.loads(line)
                fqn = content['element']['data']['fqn']
                cache.add(fqn)
    
    source_annotated = source.content_utf8
    source_wo_proof = source.content_utf8
    to_do: List[Tuple[int, str, VernacElement]] = []
    target_elements, proof_elements = parser.ast(source)
    for element in target_elements:
        to_do.append((element.span.bp, "COMMENT", element))
    for element in proof_elements:
        to_do.append((element.span.bp, "DELETE", element))

    to_do.sort(reverse=True)
    fqn_vernac_element: Dict[str, VernacElement] = {}
    for _, task, element in to_do:
        span = element.span
        if task == 'COMMENT':
            if element.kind == VernacKind.SYNTACTIC_DEFINITION or element.kind == VernacKind.NOTATION:
                fqn = element.data['history_name']
            else:
                fqn = element.data['fqn']
            assert fqn not in fqn_vernac_element, f"Issue with fqn (already met): {fqn} in {source.path}"

            fqn_vernac_element[fqn] = element
            instr_annotation = f"\n<annotation>{fqn}</annotation>\n".encode('utf-8')
            source_annotated = source_annotated[:span.bp] + instr_annotation + source_annotated[span.bp:]
        elif task == 'DELETE':
            eps = 0
            if source_annotated.decode('utf-8')[span.ep] == '\n':
                eps = 1
            source_annotated = source_annotated[:span.bp] + source_annotated[span.ep+eps:]
            source_wo_proof = source_wo_proof[:span.bp] + source_wo_proof[span.ep+eps:]
    
    source_annotated = remove_ocaml_comments(source_annotated.decode('utf-8'))
    source_wo_proof = source_wo_proof.decode('utf-8')

    with open('test_annotated.v', 'w') as file:
        file.write(source_annotated)
    with open('test_wo_proof.v', 'w') as file:
        file.write(source_wo_proof)
    exit()
    return source_wo_proof, source_annotated, fqn_vernac_element

try:
    lib_path = '/home/theo/.opam/mc_dev/lib/coq/user-contrib/mathcomp/'
    to_do = []
    for dirpath, _, filenames in os.walk(lib_path):
        for filename in filenames:
            if not filename.endswith('.v'):
                continue

            filepath = os.path.join(dirpath, filename)
            source = Source.from_local_path(filepath)
            to_do.append(source)
    
    for source in tqdm(to_do):
        do_task(source)
finally:
    if opam_docker is not None:
        opam_docker.close()