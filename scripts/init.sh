# Install Python version 3.9.13 with pyenv
pyenv install 3.9.13

# Create a new virtual environment called "moviescripts" with Python 3.9.13
pyenv virtualenv 3.9.13 moviescripts

# Set the local Python version to be the one specified in the "moviescripts" virtual environment
pyenv local moviescripts

# Activate the "moviescripts" virtual environment
pyenv activate moviescripts

# Install the package manager Poetry
pip install poetry

# Install dependencies specified in the Poetry lock file
poetry install

# Run the "train" command specified in the Poetry project
poetry run train
