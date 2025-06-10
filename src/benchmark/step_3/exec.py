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
    return json.loads(completion.choices[0].message.content)['query']

def process_prompt(prompt, export_path, data, client, config, delay=0):
    """
    Executes generation according to prompt
    """
    time.sleep(delay)
    output_json = generate_output(prompt, client, config)
    data['query'] = output_json
    with open(export_path, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/benchmark/step_2/result.json', help='Input path')
    parser.add_argument('--context-dir', default='export/output/step_1_bis/', help='Input path')
    parser.add_argument('--output', default='export/benchmark/step_3', help='Output dataset path')
    parser.add_argument('--config-dir', default='config/benchmark/step_3')
    parser.add_argument('--max-workers', default=100, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean-delay', default=10, type=int, help='Mean delay before a request is send: use this parameter to load balance')


    args = parser.parse_args()
    config_path = os.path.join(args.config_dir, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    prompt_path = os.path.join(args.config_dir, f'prompt.txt')
    with open(prompt_path, 'r') as file:
        prompt_template = file.read()
    client = OpenAI(
        base_url=config['base_url'],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    to_do = []

    with open(args.input, 'r') as file:
        benchmark_content = json.load(file)
    
    for benchmark_kind in benchmark_content:
        os.makedirs(os.path.join(args.output, benchmark_kind), exist_ok=True)
        for entry in benchmark_content[benchmark_kind]:
            
            step_str = [c for c,_,_ in entry['steps']]
            steps = "\n".join(step_str[:-1])
            last_step = step_str[-1]

            last_constant = random.choice(entry['steps'][-1][1][benchmark_kind])
            fullname = last_constant[2]
            docstring = last_constant[3]

            parent = entry['parent']
            element_name = entry['relative_name']

            context_path = os.path.join(args.context_dir, f'docstring#{parent.split('.')[-1]}.v')
            with open(context_path, 'r') as file:
                context = file.read()
            constant_parent = last_constant[0]
            constant_relative_name = last_constant[1]
            constant_fqn = f'{constant_parent}.{constant_relative_name}'
            entry['query_constant'] = {'parent': constant_parent, 'relative_name': constant_relative_name, 'fqn': constant_fqn}
            prompt = prompt_template.format(context=context, statement=entry['fullname'], steps=steps, next_step=last_step, fullname=fullname)
            export_path = os.path.join(args.output, benchmark_kind, f'term_{parent.replace('.', '_')}_{element_name.replace('.', '_')}.json')
            if not os.path.exists(export_path):
                to_do.append((prompt, export_path, entry))
    delay_max = args.mean_delay*2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:  # Adjust the number of workers as needed
        futures = [executor.submit(process_prompt, prompt, export, entry, client, config['request_config'], delay=random.randint(0, delay_max)) for prompt, export, entry in to_do]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
