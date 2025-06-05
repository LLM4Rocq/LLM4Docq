import re
import os
import argparse
import shutil
import json
from collections import defaultdict

class NameNotFound(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def extract_entries(source_code, delta_line=0):
    # Merging comments
    source_code = re.sub(r'\*\)\s*\(\*', '', source_code)
    source_code = source_code + '\n' # need to add an extra \n to manage edge case 
    pattern = (
        r'(Lemma|Definition|Notation|Fact|Theorem|Record|Fixpoint)\s+'  # capture kind
        r'((\".+?\")|(\S+)).*?\.\s*\n'                  # capture name (either quoted or bare), up to “.\n”
    )
    matches = re.finditer(pattern, source_code, flags=re.DOTALL)

    result = {}
    for match in matches:
        start_idx = match.start()
        end_idx   = match.end()

        # count how many newlines appear before each index:
        #   line numbers are 1-based, so add 1
        start_line = source_code.count('\n', 0, start_idx) + 1
        end_line   = source_code.count('\n', 0, end_idx) + 1

        # docstring = match.group(1)
        kind      = match.group(1)
        name      = match.group(2)
        fullname  = match.group(0)

        if name != '_':  
            result[name] = {
                "name":       name,
                "kind":       kind,
                "docstring":  "",
                "fullname":   remove_comments(fullname).strip(),
                "start_line": start_line + delta_line,
                "end_line":   end_line + delta_line
            }

    return result


def read_modules(content: str) -> list:
    """Split some content module by module."""

    match = re.search(r"\sModule\s(|Export\s|Import\s)(?P<name>[_'a-zA-Z0-9]*)\.\s", content)

    if match:
        result = [content[:match.start()+1]]

        module_name = match.group("name")
        module_start = content[match.start()+1:match.end()]
        close_module = f"End {module_name}."
        content = content[match.end()-1:]
        close_idx = content.find(close_module)
        if close_idx < 0:
            raise Exception(f"Error: the module {module_name} is not closed.")

        module_content = read_modules(content[:close_idx])
        module_content[0] = module_start + module_content[0]
        module_content[-1] += f"End {module_name}."

        result.append((module_name, module_content))
        content = content[close_idx+len(close_module):]
        result += read_modules(content)
        return result

    else:
        return [content]

def remove_comments(content: str) -> str:
    """Remove Rocq comments in a file."""
    pattern = r"\(\*[\s\S]*?\*\)"
    match = re.search(pattern, content, flags=re.DOTALL)
    while match:
        content = content[:match.start()] + content[match.end():]
        match = re.search(pattern, content)
    return content

def flatten_modules(source, parent=""):
    if isinstance(source, str):
        return [(parent, source.strip())]
    elif isinstance(source, tuple):
        module_name, subcontent = source
        if parent:
            new_parent = parent +'.'+ module_name
        else:
            new_parent = module_name
        return flatten_modules(subcontent, parent=new_parent)
    else:
        result = []
        for subcontent in source:
            result += flatten_modules(subcontent, parent=parent)
        return result

def extract_skeleton(content:str):
    modules = read_modules(content)
    flat_m = flatten_modules(modules)
    
    concat_source = "".join([source for _,source in flat_m])
    result = {}
    delta_line = 0
    for parent, source in flat_m:
        for entry in extract_entries(source, delta_line=delta_line).values():
            name = entry['name']
            fqn = f'{parent}.{name}' if parent else name
            result[fqn] = entry
        delta_line += len(source.split('\n'))-1
    return concat_source, result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess library for next steps.")
    parser.add_argument("--library-dir", default="export/output/step_0", help="Directory of preprocess library")
    parser.add_argument("--export-dir", default="export/output/step_1")
    args = parser.parse_args()

    output = {}
    stats = defaultdict(lambda:0)
    shutil.copytree(args.library_dir, args.export_dir, dirs_exist_ok=True)

    for (root,dirs,files) in os.walk(args.export_dir, topdown=True):
        for file in files:
            if file.endswith('.v'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file:
                    content = file.read()
                new_source, skeleton = extract_skeleton(content)
                for entry in skeleton.values():
                    stats[entry['kind']] += 1
                relfilepath = os.path.relpath(filepath, args.export_dir)
                relfilepath = relfilepath.removesuffix('.v').replace('/', '.')
                
                output[relfilepath] = skeleton

                with open(filepath, 'w') as file:
                    file.write(new_source)
    
    for key, value in stats.items():
        print(f"{key}: {value}")

    os.makedirs(args.export_dir, exist_ok=True)
    with open(os.path.join(args.export_dir, 'result.json'), 'w') as file:
        json.dump(output, file, indent=4)