"""
1. Run: `pip install openai agno lancedb tantivy sqlalchemy` to install the dependencies
2. Export your OPENAI_API_KEY
3. Run: `python cookbook/reasoning/tools/knowledge_tools.py` to run the agent
"""

from dotenv import load_dotenv
import os

from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.ollama import Ollama
from agno.tools.knowledge import KnowledgeTools
from agno.vectordb.pgvector import PgVector, SearchType

# Load environment variables from .env
load_dotenv()

# Use environment variable for database URL
db_url = os.getenv("DB_URL")

# Create a knowledge base containing information from a URL
agno_docs = UrlKnowledge(
     urls=[
        "https://r.jina.ai/https://wallety.cash",
        "https://r.jina.ai/https://wallety.cash/company/"
    ],    
    vector_db=PgVector(
        table_name="wallety_assist_knowledge",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OllamaEmbedder(id="llama2:7b")
    ),
)

knowledge_tools = KnowledgeTools(
    knowledge=agno_docs,
    think=True,
    search=True,
    analyze=True,
    add_few_shot=True,
)

agent = Agent(
    model=Ollama(id="llama3.1:8b"),
    tools=[knowledge_tools],
    show_tool_calls=True,
    markdown=True,
)

if __name__ == "__main__":
    # Load the knowledge base, comment after first run
    # agno_docs.load(recreate=True)
    agent.print_response("What services do I have access to while using Wallety?", stream=True)
