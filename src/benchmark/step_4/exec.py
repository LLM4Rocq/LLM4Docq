import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/benchmark/step_3', help='Input path')
    parser.add_argument('--output', default='export/benchmark/step_4', help='Output dataset path')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for benchmark_name in os.listdir(args.input):
        benchmark_path = os.path.join(args.input, benchmark_name)
        result = []
        for term_name in os.listdir(benchmark_path):
            term_path = os.path.join(benchmark_path, term_name)
            with open(term_path, 'r') as file:
                term = json.load(file)
            result.append(term)
        
        export_path = os.path.join(args.output, f'result_{benchmark_name}.json')
        with open(export_path, 'w') as file:
            json.dump(result, file, indent=4)
