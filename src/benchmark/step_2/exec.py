import argparse
import os
import sys
import re
import json
from copy import deepcopy

# to avoid issue with json recursion
sys.setrecursionlimit(10_000) 

import numpy as np
import bm25s
from tqdm import tqdm


def select_diverse_documents(documents, filepaths, k):
    """
    Extracts subset of diverse documents using BM25.
    """
    # Not efficient, but enough for the moment
    retriever = bm25s.BM25(corpus=documents)
    retriever.index(bm25s.tokenize(documents))
    similarity_matrix = np.zeros((len(documents), len(documents)))
    
    for i, doc in tqdm(enumerate(documents)):
        tokenized_doc = bm25s.tokenize(doc)
        scores = retriever.retrieve(tokenized_doc, k=len(documents)).scores
        similarity_matrix[i, :] = scores

    selected_indices = [0]  # Start with the first document
    while len(selected_indices) < k:
        min_similarities = []
        
        for i in range(len(documents)):
            if i not in selected_indices:
                min_sim = min(similarity_matrix[i, selected_indices])
                min_similarities.append((i, min_sim))
        next_doc = min(min_similarities, key=lambda x: x[1])[0]
        selected_indices.append(next_doc)
    return [filepaths[i] for i in selected_indices]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/benchmark/step_1', help='Output path previous step')
    parser.add_argument('--output', default='export/benchmark/step_2/', help='New output path')
    parser.add_argument('--num-documents', default=200, help='Maximum number of final documents')

    args = parser.parse_args()


    elements_outside = []
    documents_outside = []

    elements_inside = []
    documents_inside = []

    num_valid = 0

    for (root,dirs,files) in os.walk(args.input, topdown=True):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file:
                    element = json.load(file)
                element_candidate_inside = deepcopy(element)
                element_candidate_outside = deepcopy(element)
                has_inside = False
                has_outside = False

                current_proof = []
                for step, constants, val in element_candidate_inside['steps']:
                    current_proof.append((step, constants, val))
                    if val and constants['current_file']:
                        has_inside = True
                        element_candidate_inside['steps'] = current_proof
                        break

                current_proof = []
                for step, constants, val in element_candidate_outside['steps']:
                    current_proof.append((step, constants, val))
                    if val and constants['outside_file']:
                        has_outside = True
                        element_candidate_outside['steps'] = current_proof
                        break
                
                if has_inside:
                    elements_inside.append(element_candidate_inside)
                    documents_inside.append(element['docstring'])
                if has_outside:
                    elements_outside.append(element_candidate_outside)
                    documents_outside.append(element['docstring'])
    
    theorems_to_keep_inside = select_diverse_documents(documents_inside, elements_inside, args.num_documents)
    theorems_to_keep_outside = select_diverse_documents(documents_outside, elements_outside, args.num_documents)
    os.makedirs(args.output, exist_ok=True)

    result = {'current_file': theorems_to_keep_inside, 'outside_file': theorems_to_keep_outside}
    with open(os.path.join(args.output, 'result.json'), 'w') as file:
        json.dump(result, file, indent=4)

