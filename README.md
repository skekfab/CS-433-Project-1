# CS-433 Project 1

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
