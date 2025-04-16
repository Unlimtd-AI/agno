# ------------------------------- SETUP PYTHON ENVIRONMENT -------------------------------
# Setup your virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install the required dependencies:
uv pip install -U agno openai duckduckgo-search elevenlabs sqlalchemy 'fastapi[standard]' lancedb pylance tantivy pandas numpy

# ------------------------------- SETUP AGNO ---------------------------------------------
# python3 -m venv ~/.venvs/agno
# source ~/.venvs/agno/bin/activate

# pip install -U agno

# pip install -U agno --no-cache-dir

# ag setup