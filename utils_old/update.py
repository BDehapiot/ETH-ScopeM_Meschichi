#%% Imports -------------------------------------------------------------------

import yaml
from pathlib import Path

#%% Initialize ----------------------------------------------------------------

root_path = Path(__file__).resolve().parents[1]
repo_name = root_path.name

#%% Read files ----------------------------------------------------------------

# Read environment.yml
with open(Path(root_path / "environment.yml"), 'r') as file:
    environment = yaml.safe_load(file)

# Read README_description.md
with open(Path(root_path / "utils" / "README_description.md"), "r") as file:
    description = file.read()

# Read README_installation.md
with open(Path(root_path / "utils" / "README_installation.md"), "r") as file:
    installation = file.read()

# Read README_template.md
with open(Path(root_path / "utils" / "README_template.md"), "r") as file:
    template = file.read()

#%% Extract relevant informations ---------------------------------------------

# Extract environment name
env_name = environment.get('name', '')

# Extract conda and pip dependencies
conda_dependencies = []
pip_dependencies = []
for dependency in environment.get('dependencies', []):
    if isinstance(dependency, str):
        conda_dependencies.append(dependency)
        if dependency.startswith('python='):
            python_version = dependency.split('=')[1]
    elif isinstance(dependency, dict) and 'pip' in dependency:
        pip_dependencies.extend(dependency['pip'])

#%% Format and replace --------------------------------------------------------

# Format conda and pip dependencies
conda_dependencies_str = ""
for dependency in conda_dependencies:
    conda_dependencies_str += f"- {dependency}\n"
pip_dependencies_str = ""
for dependency in pip_dependencies:
    pip_dependencies_str += f"- {dependency}\n"

# Replace placeholders
installation = installation.replace("{{ repo_name }}", repo_name)
installation = installation.replace("{{ env_name }}", env_name)
template = template.replace("{{ repo_name }}", repo_name)
template = template.replace("{{ env_name }}", env_name)
template = template.replace("{{ python_version }}", python_version)
template = template.replace("{{ description }}", description)
template = template.replace("{{ installation }}", installation)
template = template.replace("{{ conda_dependencies }}", conda_dependencies_str)
template = template.replace("{{ pip_dependencies }}", pip_dependencies_str)

#%% Update README -------------------------------------------------------------

# Write README.md in root path
with open(Path(root_path / "README.md"), "w") as file:
    file.write(template)