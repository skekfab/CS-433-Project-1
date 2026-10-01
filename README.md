# CS-433 Project 1

[![Overleaf Paper](https://img.shields.io/badge/Overleaf_Paper-Open-green)](hhttps://www.overleaf.com/project/6abea0d8f69d58b37f0a9b2e)
&nbsp;&nbsp;
[![Grading Tests](https://github.com/skekfab/CS-433-Project-1/actions/workflows/grading.yml/badge.svg)](https://github.com/skekfab/CS-433-Project-1/actions/workflows/grading.yml)

## Setup and Usage
```zsh
# [Optional] if you're using pyenv
pyenv local 3.14
poetry env use $(which python3)

# Install required packages (including test dependencies for development)
poetry install
poetry install --with test
```

## Usage
```zsh
# To add a new library
poetry add <library-name>

# To apply the changes of pyproject.toml
poetry lock

# Run the grading tests
pytest --github_link . grading_tests/
```
