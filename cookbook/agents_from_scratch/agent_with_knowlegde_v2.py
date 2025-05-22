import os

from dotenv import load_dotenv

from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.models.ollama import Ollama
from agno.storage.postgres import PostgresStorage
from agno.vectordb.pgvector import PgVector, SearchType
from agno.tools.eleven_labs import ElevenLabsTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase, PDFUrlReader

# Load environment variables from .env
load_dotenv()

# Use environment variable for database URL
db_url = os.getenv("NJM_DB_URL")

# Initialize knowledge & storage
agent_knowledge = PDFUrlKnowledgeBase(
    urls=["https://visibillfiles.blob.core.windows.net/visibill/Visibill/Images/the_way_of_the_superior_man.pdf"],    
    vector_db=PgVector(
      table_name="superior_man_knowledge",
      db_url=db_url,
      search_type=SearchType.hybrid,
      embedder=OllamaEmbedder(id="llama2:70b"),
    ),
    reader=PDFUrlReader()
)

agent_storage = PostgresStorage(
    # store sessions in the ai.sessions table
    table_name="superior_man_sessions",
    # db_url: Postgres database URL
    db_url=db_url,
)

audio_agent = Agent(
    model=Ollama(id="llama3.1:70b"),
    tools=[
      ElevenLabsTools(
         voice_id="txeQP6PnLIaeZ8sagUIz",
         model_id="eleven_multilingual_v2",
         target_directory="audio_generations"
      )
   ],
    description="You are an AI agent that can generate audio using the ElevenLabs API.",
    instructions=[
      "Use the pdf provided to generate audio.",
      # "The user can say generate audio for the whole of Chapter 1. You will provide audio only for that chapter.",
      "When the user asks you to generate audio, use the `text_to_speech` tool to generate the audio.",
      "You'll generate the appropriate prompt to send to the tool to generate audio.",
      "You don't need to find the appropriate voice first, I already specified the voice to use.",
      "Return the audio file name in your response. Don't convert it to markdown.",
      "The audio should be long and detailed.",
      "The audio must be at least 3 minutes long. That means at least 800 words of content must be sent to the text_to_speech tool. Ensure your output has that much text before calling the tool."
    ],
    
    knowledge=agent_knowledge,
    storage=agent_storage,
    
    markdown=True,
    debug_mode=True,
    show_tool_calls=True,
    
    # 1. Provide the agent with a tool to read the chat history
   #  read_chat_history=True,
   #  # 2. Automatically add the chat history to the messages sent to the model
   #  add_history_to_messages=True,
   #  # Number of historical runs to add to the messages.
   #  num_history_responses=3,
)

if __name__ == "__main__":
   # Set to False after the knowledge base is loaded
   load_knowledge = False
   if load_knowledge:
      agent_knowledge.load()

   # audio_agent.print_response("What can I learn from the way of the superior man.")
   audio_agent.print_response("Generate a very long audio of Dealing with Women. Use the pdf knowleged you have as context.")
