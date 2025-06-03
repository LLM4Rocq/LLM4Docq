import os
import json
from collections import defaultdict
import argparse

from tqdm import tqdm

from label_studio_sdk.client import LabelStudio
from label_studio_sdk.core.request_options import RequestOptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create projects based on provided dataset.")
    parser.add_argument("--label-studio-url", default='http://localhost:8080')
    parser.add_argument("--ds-path", default="export/ds.json", help="Directory for output images")
    args = parser.parse_args()
    # Connect to the Label Studio API and check the connection
    ls = LabelStudio(base_url=args.label_studio_url, api_key=os.environ.get('API_KEY'))

    with open('src/label_studio/interface.xml', 'r') as file:
        interface = file.read()

    # Load dataset
    with open(args.ds_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_datasets = defaultdict(list)
    for parent in data:
        for element_name, element in data[parent].items():
            fqn = parent + '.' + element_name
            element['fqn'] = fqn
            all_datasets[parent].append(element)

    for key in tqdm(all_datasets):
        project_resp = ls.projects.create(title=key,enable_empty_annotation=False, label_config=interface)
        project_id = project_resp.id
        data = all_datasets[key]
        ls.projects.import_tasks(id=project_id, request=data)
        param = {"sampling":"Uniform sampling"}
        ls.projects.update(project_id, request_options=RequestOptions(additional_body_parameters=param))