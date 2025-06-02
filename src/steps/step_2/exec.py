import os
import random
import argparse
import time
import random
import json
import concurrent.futures

import yaml
from openai import OpenAI
from tqdm import tqdm

"""
Fourth step: Generate reasonings using deepseek R1.
"""

def generate_output(prompt, client, config):
    """
    Sends prompt to client using config.
    """
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        **config
    )
    return {"content": completion.choices[0].message.content}

def process_prompt(prompt, export_path, data, client, config, delay=0):
    """
    Executes generation according to prompt
    """
    time.sleep(delay)
    output_entry = generate_output(prompt, client, config)
    result = {'data': data, 'output': output_entry}
    with open(export_path, 'w') as file:
        json.dump(result, file, indent=4)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dataset', default='export/output/step_1/result.json', help='Input dataset path')
    parser.add_argument('--library-dir', default='export/output/step_1', help='Preprocess library dir')
    parser.add_argument('--output', default='export/output/step_2', help='Output dataset path')
    parser.add_argument('--config-dir', default='config/step_2')
    parser.add_argument('--prompt-dir', default='config/step_2/prompts')
    parser.add_argument('--max-workers', default=100, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean-delay', default=10, type=int, help='Mean delay before a request is send: use this parameter to load balance')

    parser.add_argument('--chunk-overlap', default=0, type=int, help='Number of lines to prepend to chunks to give some additionnal context')
    parser.add_argument('--chunk-size', default=500, type=int, help='Maximum number of lines contains in each chunk')
    parser.add_argument('--max-annotations', default=50, type=int, help='Maximum number of elements to annotate with a docstring')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config_path = os.path.join(args.config_dir, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    with open(args.input_dataset, 'r') as file:
        input_content = json.load(file)

    client = OpenAI(
        base_url=config['base_url'],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    to_do = []

    for parent, subdict in input_content.items():
        if 'matrix' not in parent:
            continue
        subname = parent.split('.')[-1]
        prompt_path = os.path.join(args.prompt_dir, f'prompt_{subname}.txt')
        if not os.path.exists(prompt_path):
            prompt_path = os.path.join(args.prompt_dir, 'prompt.txt')
        with open(prompt_path, 'r') as file:
            prompt_template = file.read()
        source_path = os.path.join(args.library_dir, parent.replace('.','/')+'.v')
        with open(source_path, 'r') as file:
            source_content = file.read()
        source_lines = source_content.split('\n')
        subdict_items = sorted(list(subdict.items()), key=lambda x: x[1]['end_line'])
        subdict_idx = 0
        num_lines = 0
        start_line = 0
        end_line = 0

        splits = []

        chunk_data = []
        while end_line < len(source_lines) and subdict_idx < len(subdict_items):

            if end_line - start_line >= args.chunk_size or len(chunk_data) >= args.max_annotations:
                splits.append((source_lines[start_line:end_line], chunk_data))
        
                start_line = end_line - args.chunk_overlap
                chunk_data = []
            
            end_line += 1
            next_entry = subdict_items[subdict_idx]
            if next_entry[1]['end_line']<= end_line:
                chunk_data.append(next_entry)
                subdict_idx += 1

        if chunk_data:
            splits.append((source_lines[start_line:end_line], chunk_data))

        for k, (chunk, chunk_data) in enumerate(splits):
            chunk = "\n".join(chunk)
            missing = "\n".join([entry[1]['name'].split('.')[-1] for entry in chunk_data])
            prompt = prompt_template.format(**{"source":chunk, "missing":missing})

            export_path = os.path.join(args.output, parent+f'#chunk_{k}')
            to_do.append((prompt, export_path, chunk_data))
        # break

    delay_max = args.mean_delay*2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:  # Adjust the number of workers as needed
        futures = []
        futures += [executor.submit(process_prompt, prompt, export, entry, client, config['request_config'], delay=random.randint(0, delay_max)) for prompt, export, entry in to_do]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
