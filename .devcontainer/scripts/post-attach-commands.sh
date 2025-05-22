# ------------------------------- SETUP PYTHON ENVIRONMENT -------------------------------
# Setup your virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install the required dependencies:
# uv pip install -U agno openai duckduckgo-search elevenlabs sqlalchemy 'fastapi[standard]' lancedb pylance tantivy pandas numpy huggingface_hub ollama mcp pydantic>=2.9.0 pydantic_core==2.14.6

uv pip install -r requirements.txt

# ------------------------------- SERVE OLLAMA ---------------------------------------------
nohup ollama serve > ollama.log 2>&1 &

echo "Pull Models"
sleep 2

ollama pull deepseek-r1:8b
ollama pull deepseek-r1:70b

# For tools
ollama pull llama3.1:8b
ollama pull llama3.1:70b

# For embedding
ollama pull llama2:7b
 ollama pull llama2:70b
 