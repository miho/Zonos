# fix_json_syntax.py
import re

def convert_json_to_python_syntax(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace JSON booleans with Python booleans
    content = re.sub(r'\btrue\b', 'True', content)
    content = re.sub(r'\bfalse\b', 'False', content)
    content = re.sub(r'\bnull\b', 'None', content)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Fixed JSON syntax in {filename}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        convert_json_to_python_syntax(sys.argv[1])
    else:
        print("Usage: python fix_json_syntax.py <filename>")