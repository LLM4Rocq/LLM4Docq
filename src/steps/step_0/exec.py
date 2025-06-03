import re
import os
import argparse
import json
import shutil
from collections import defaultdict


def remove_proofs(content: str) -> str:
    pattern = r'Proof\.(.*?)(Qed\.|Abort\.)'
    to_do = []
    for match in re.finditer(pattern, content, flags=re.DOTALL):
        start, end = match.start(0), match.end(0)
        to_do.append((start, end))
    to_do = sorted(to_do, reverse=True)
    for start, end in to_do:
        content = content[:start] + content[end:]
    return content

def remove_comments(content: str) -> str:
    """Remove Rocq comments in a file."""
    pattern = r"\(\*[\s\S]*?\*\)"
    match = re.search(pattern, content, flags=re.DOTALL)
    while match:
        content = content[:match.start()] + content[match.end():]
        match = re.search(pattern, content)
    return content

def remove_blank(content: str) -> str:
    while '\n\n\n' in content:
        content = content.replace('\n\n\n', '\n\n')
    return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse library to extract modules skeleton.")
    parser.add_argument("--library-dir", default="export/mathcomp/", help="Directory for output images")
    parser.add_argument("--export-dir", default="export/output/step_0")
    args = parser.parse_args()

    shutil.copytree(args.library_dir, args.export_dir, dirs_exist_ok=True)
    output = {}
    stats = defaultdict(lambda:0)
    for (root,dirs,files) in os.walk(args.export_dir, topdown=True):
        for file in files:
            if file.endswith('.v'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file:
                    content = file.read()
                
                assert (content.count('Proof.') - content.count('Qed.') - content.count('Abort.'))==0, f"Issue, source file {filepath} contains proofs that are not well contained in a Proof.[..]Qed. block"

                content = remove_proofs(content)
                content = remove_comments(content)
                content = remove_blank(content)
                with open(filepath, 'w') as file:
                    file.write(content)