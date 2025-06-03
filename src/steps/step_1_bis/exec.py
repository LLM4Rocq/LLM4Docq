import os
import random
import argparse
import time
import random
import concurrent.futures
import re

import yaml
from openai import OpenAI
from tqdm import tqdm

class NoCodeFound(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


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

def extract_code(content: str):
    pattern = r"```code(.*)```"
    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        raise NoCodeFound(f"No code found in {content}")
    return match.group(1)

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
    return extract_code(completion.choices[0].message.content)

def process_prompt(prompt, export_path, export_prompt_path, prompt_export_template, client, config, delay=0, max_retry=3):
    """
    Executes generation according to prompt
    """
    time.sleep(delay)
    for k in range(max_retry):
        try:
            output = generate_output(prompt, client, config)
            with open(export_path, 'w') as file:
                file.write(output)
            with open(export_prompt_path, 'w') as file:
                output_adapt = output.replace('{', '{{')
                output_adapt = output_adapt.replace('}', '}}')
                file.write(prompt_export_template.format(docstring=output_adapt))
            break
        except NoCodeFound as e:
            with open(f"{export_path}_error_{k}", 'w') as file:
                file.write(e.message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--library-dir', default='export/mathcomp', help='Library dir')
    parser.add_argument('--output', default='export/output/step_1_bis', help='Output dataset path')
    parser.add_argument('--config-dir', default='config/step_1_bis')
    parser.add_argument('--export-prompt-path', default='config/step_2/prompts/', help='Directory to export prompt')
    parser.add_argument('--max-retry', default=3, type=int, help='Max number of retry before having a correct code block')
    parser.add_argument('--max-workers', default=100, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean-delay', default=10, type=int, help='Mean delay before a request is send: use this parameter to load balance')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config_path = os.path.join(args.config_dir, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    prompt_export_path = os.path.join(args.config_dir, 'prompt_export.txt')
    with open(prompt_export_path, 'r') as file:
        prompt_export_template = file.read()

    client = OpenAI(
        base_url=config['base_url'],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    to_do = []
    prompt_template_path = os.path.join(args.config_dir, 'prompt.txt')
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()

    for (root,dirs,files) in os.walk(args.library_dir, topdown=True):
        for file in files:
            if file.endswith('.v'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as fileio:
                    source = fileio.read()
                source = remove_proofs(source)
                prompt = prompt_template.format(**{"source":source})
                export_path = os.path.join(args.output, f'docstring#{file}')
                export_prompt_path = os.path.join(args.export_prompt_path, f'prompt_{file.removesuffix('.v')}.txt')
                if not os.path.exists(export_path):
                    to_do.append((prompt, export_path, export_prompt_path))
    delay_max = args.mean_delay*2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:  # Adjust the number of workers as needed
        futures = []
        futures += [executor.submit(process_prompt, prompt, export_path, export_prompt_path, prompt_export_template, client, config['request_config'], delay=random.randint(0, delay_max), max_retry=args.max_retry) for prompt, export_path, export_prompt_path in to_do]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass