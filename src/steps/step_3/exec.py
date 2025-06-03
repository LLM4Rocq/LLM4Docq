import re
import os
import argparse
import json
from collections import defaultdict

from Levenshtein import distance

class NoJsonFound(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def extract_json_code(content: str):
    pattern = r"```json(.*)```"
    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        raise NoJsonFound(f"No json found in {content}")
    return json.loads(match.group(1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess output.")
    parser.add_argument("--input", default="export/output/step_2", help="Directory of output")
    parser.add_argument("--tolerance", default=4, type=int, help="Maximum Levenshtein distance tolerate between name in output json, and name in source entries")
    parser.add_argument("--export-dir", default="export/output/step_3")
    args = parser.parse_args()

    stats = defaultdict(lambda:0)
    result_list = defaultdict(list)
    result = defaultdict(dict)
    for (root,dirs,files) in os.walk(args.input, topdown=True):
        for file in files:
            if "#chunk" in file:
                print(file)
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as fileio:
                    content = json.load(fileio)

                parent = file.split('#chunk')[0]

                chunk_data = content['data']
                output = content['output']
                for entry_data, entry_output in zip(chunk_data, output):
                    name_data = entry_data[1]['name']
                    name_output = entry_output['name']
                    assert distance(name_output, name_data) <= args.tolerance, f"Issue with {entry_output}, {name_output}"

                    entry_data[1]['docstring'] = entry_output['docstring']
                    stats[entry_data[1]['kind']] += 1
                result_list[parent] += chunk_data

    for parent in result_list:
        result_list[parent].sort(key=lambda x:x[1]['end_line'])
        result[parent] = result[parent] | dict(result_list[parent])

    for key, value in stats.items():
        print(f"{key}: {value}")

    os.makedirs(args.export_dir, exist_ok=True)
    with open(os.path.join(args.export_dir, 'result.json'), 'w') as file:
        json.dump(result, file, indent=4)