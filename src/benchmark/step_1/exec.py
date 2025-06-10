import argparse
import os
import sys
import re
import json
from collections import defaultdict
import concurrent.futures
import logging
logger = logging.getLogger(__name__)

# to avoid issue with json recursion
sys.setrecursionlimit(10_000) 

from tqdm import tqdm

from src.training.eval import eval_tactics, start_pet_server, stop_pet_server, timeout, TimeoutError

def extract_constants(content: str, constants_dict: dict):
    pattern = r'[a-zA-Z0_9\.@_]*'
    matches = re.finditer(pattern, content, flags=re.DOTALL)
    result = []
    for match in matches:
        constant = match.group(0)
        if constant in constants_dict:
            result.append(constants_dict[constant])
    return result

def extract_steps(content: str):
    pattern = r'(.*?\.)\s'
    result = []
    matches = re.finditer(pattern, content + ' ', flags=re.DOTALL)
    for match in matches:
        step = match.group(0).strip()
        if step.startswith('-'):
            result.append('-')
            step = step[1:]
        if step.startswith('+'):
            result.append('+')
            step = step[1:]
        result.append(step.strip())
    return result

def check_list(data_list, port=8765):
    server_process = start_pet_server(mean_wait=1, port=port)
    first_eval_tactics = timeout(seconds=60)(eval_tactics)
    second_eval_tactics = timeout(seconds=10)(eval_tactics)
    first_tactic = True
    for k, (data, data_path) in list(enumerate(data_list)):
        name_thm = data['name']
        workspace = os.path.abspath(data['workspace'])
        filepath = data['filepath']
        tactics = [s[0] for s in data['steps']]
        try:
            if first_tactic:
                goal_init, res = first_eval_tactics(name_thm, workspace, filepath, tactics, port=port)
                first_tactic = False
            else:
                goal_init, res = second_eval_tactics(name_thm, workspace, filepath, tactics, port=port)
        except TimeoutError as e:
            print(e)
            stop_pet_server(server_process)
            server_process = start_pet_server(mean_wait=1, port=port)
            first_tactic = True
            continue

        if res[-1]['status'] != 'finish':
            logger.warning(f'{data_path} does not compile using Pytanque. Ignore the file.')
            logger.warning(f'Last result: {res[-1]}')
            continue
        
        data['evaluation'] = res
        data['goals'] = [goal_init] + [entry['goals'] for entry in res]
        data['pytanque_check'] = True
        with open(data_path, 'w') as file_io:
            json.dump(data, file_io, indent=4)
        if k%1000 == 999:
            stop_pet_server(server_process)
            server_process = start_pet_server(mean_wait=1, port=port)
            first_tactic = True
    stop_pet_server(server_process)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dataset-elements', default='export/output/step_1/result.json', help='Output path previous step')
    parser.add_argument('--input-dataset-statement', default='export/benchmark/step_0/result.json', help='Output path previous step')
    parser.add_argument('--output', default='export/benchmark/step_1/', help='New output path')
    parser.add_argument('--num-documents', default=200, help='Maximum number of final documents')
    parser.add_argument('--workspace-dir', default='export/mathcomp/')
    parser.add_argument('--max-workers', default=8, type=int, help='Number of workers')

    args = parser.parse_args()

    with open(args.input_dataset_elements, 'r') as file:
        content = json.load(file)
    with open(args.input_dataset_statement, 'r') as file:
        content_statement = json.load(file)

    os.makedirs(args.output, exist_ok=True)
    constants_dict = {}
    duplicate = set()
    for parent in content:
        for element_name in content[parent]:
            name = content[parent][element_name]['name']
            if name not in constants_dict and name not in duplicate:
                element = content[parent][element_name]
                element['parent'] = parent
                element['relative_name'] = element_name
                constants_dict[name] = element
                
            else:
                if name in constants_dict:
                    del constants_dict[name]
                duplicate.add(name)
    result = defaultdict(dict)
    to_do = defaultdict(list)
    proof_current = []
    proof_outside = []
    proof_to_keep = []
    documents = []
    NUM_MIN_STEPS = 3
    MAX_CONSTANTS = 1

    num_valid = 0
    for parent in content:
        for k, element_name in enumerate(content_statement[parent]):
            element = content_statement[parent][element_name]
            proof = element['proof']
            
            steps = extract_steps(proof)
            num_current = 0
            num_outside = 0
            res_steps = []
            overall_valid = False
            for step in steps:
                constants = extract_constants(step, constants_dict)
                premises = {'current_file': [], 'outside_file': []}
                for c in constants:
                    if len(c['name']) < 4:
                        continue
                    if c['parent'] != parent:
                        premises['outside_file'].append((c['parent'], c['relative_name'], c['fullname'])) # to avoid circular reference, we can't point to c
                    else:
                        premises['current_file'].append((c['parent'], c['relative_name'], c['fullname']))
                is_valid = 1 <= len(premises['current_file']) + len(premises['outside_file']) <= MAX_CONSTANTS
                overall_valid = overall_valid or is_valid
                num_current += len(premises['current_file'])
                num_outside += len(premises['outside_file'])
                res_steps.append((step, premises, is_valid))
            
            source_path = os.path.join(args.workspace_dir, parent.replace('.','/')) + '.v'
            export_path = os.path.join(args.output, f'term_{parent.replace('.', '_')}_{element_name.replace('.', '_')}.json')

            element['steps'] = res_steps
            element['parent'] = parent
            element['relative_name'] = element_name
            element['fqn'] = f'{parent}.{element_name}'
            element['workspace'] = args.workspace_dir
            element['filepath'] = source_path
            if len(steps) > NUM_MIN_STEPS and is_valid:
                if not os.path.exists(export_path):
                    to_do[parent].append((element, export_path))
                num_valid += 1
    
    print(f'Number of valid elements: {num_valid}')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for k, source in enumerate(to_do, start=1):
            futures.append(executor.submit(check_list, to_do[source], port=8765 + k))
        
        for _ in tqdm(concurrent.futures.as_completed(futures), desc="Overall progress", position=0, total=len(futures)):
            pass
