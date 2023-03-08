"""Generate a ZenML project for a tool"""
import argparse
import logging
import os
import shutil
from textwrap import dedent


def get_hello_world_str():
    return dedent(
        f"""\
import logging

def main():
    pass
    
if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()

"""
    )


def get_readme_str(name: str):
    return dedent(
        f"""\
# Playground for {name}
    
## Installation
```
cd {name}
poetry install
```
    """
    )


def get_flake8_str():
    return dedent(
        """\
    [flake8]
    max-line-length = 79
    max-complexity = 18
    select = B,C,E,F,W,T4,B9
    ignore = E203, E266, E501, W503, F403, F401
    """
    )


def get_project_toml_str(name: str, author: str = "Author <author@gmail.com>"):
    return dedent(
        f"""\
    [tool.poetry]
    name = "{name}"
    version = "1.0.0"
    description = "{name}"
    authors = ["{author}"]
    license = "Apache 2.0"
    
    [tool.poetry.dependencies]
    python = ">=3.7.0,<3.9.0"
    
    [tool.poetry.dev-dependencies]
    black = "^21.9b0"
    isort = "^5.9.3"
    pytest = "^6.2.5"
    
    [build-system]
    requires = ["poetry-core>=1.0.0"]
    build-backend = "poetry.core.masonry.api"
    
    [tool.isort]
    profile = "black"
    known_third_party = []
    skip_glob = []
    line_length = 79
    
    [tool.black]
    line-length = 79
    include = '\.pyi?$'
    exclude = '''
    /(
        \.git
    | \.hg
    | \.mypy_cache
    | \.tox
    | \.venv
    | _build
    | buck-out
    | build
    )/
    '''
    """
    )


def write_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tool_name", type=str, help="Name of the tool")
    args = parser.parse_args()

    path = os.path.join(os.getcwd(), args.tool_name)
    src_path = os.path.join(path, "src")
    if os.path.exists(path):
        raise AssertionError(f"{path} already exists!")

    toml_str = get_project_toml_str(args.tool_name)
    flake8_str = get_flake8_str()
    py_str = get_hello_world_str()
    readme_str = get_readme_str(args.tool_name)

    # make dirs
    os.mkdir(path)
    os.mkdir(src_path)

    # copy .gitignore
    shutil.copy(
        os.path.join(os.getcwd(), ".gitignore"),
        os.path.join(path, ".gitignore"),
    )

    # write files
    write_file(os.path.join(path, ".flake8"), flake8_str)
    write_file(os.path.join(src_path, "main.py"), py_str)
    write_file(os.path.join(path, "pyproject.toml"), toml_str)
    write_file(os.path.join(path, "README.md"), readme_str)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
