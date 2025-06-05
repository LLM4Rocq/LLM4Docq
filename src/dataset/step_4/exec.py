import json
import argparse

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export dataset.")
    parser.add_argument("--input", default="export/output/step_3/result.json", help="Directory of output")
    parser.add_argument("--output", default="export/ds.json")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize Coq lexer and HTML formatter
    lexer = get_lexer_by_name("coq")
    formatter = HtmlFormatter(cssclass="highlight", nowrap=True)
    css = HtmlFormatter(cssclass="highlight").get_style_defs(".highlight")

    # Process each entry
    for file in data:
        for element in data[file]:
            code = data[file][element]['fullname']
            highlighted_code = highlight(code, lexer, formatter)
            data[file][element]['html'] = f'<style>{css}</style><pre><code class="highlight">{highlighted_code}</code></pre>'

    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
