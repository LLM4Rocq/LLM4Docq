import json
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

# Load your dataset
with open('export_bis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize Coq lexer and HTML formatter
lexer = get_lexer_by_name("coq")
formatter = HtmlFormatter(cssclass="highlight", nowrap=True)
css = HtmlFormatter(cssclass="highlight").get_style_defs(".highlight")

# Process each entry
for entry in data:
    code = entry['data']['fullname']
    highlighted_code = highlight(code, lexer, formatter)
    # Wrap in <pre><code> tags with Coq language class
    entry['data']['html'] = f'<style>{css}</style><pre><code class="highlight">{highlighted_code}</code></pre>'

# Save the modified dataset
data = {"test0":data[:3], "test1": data[4:10], "test2": data[11:20]}
for entry in data:
  with open(f'output_dataset_{entry}.json', 'w', encoding='utf-8') as f:
      json.dump(data[entry], f, indent=2, ensure_ascii=False)
