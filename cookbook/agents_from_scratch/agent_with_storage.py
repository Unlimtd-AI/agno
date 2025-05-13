"""Agent with Storage - An agent that can store sessions in a database

Install dependencies: `pip install openai lancedb tantivy sqlalchemy agno`
"""

from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.ollama import Ollama
from agno.storage.postgres import PostgresStorage
from agno.vectordb.pgvector import PgVector, SearchType

db_url = "postgresql+psycopg://postgres:q4ADyD7fMZkJ3L1v95J6@pg15-prod-rw.wallety.pg.db-eu2.zcloud.ws:60421/wallety"

# Initialize knowledge & storage
agent_knowledge = UrlKnowledge(
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

agent_storage = PostgresStorage(
    # store sessions in the ai.sessions table
    table_name="wallety_assist_sessions",
    # db_url: Postgres database URL
    db_url=db_url,
)

agent_with_storage = Agent(
    name="Agent with Knowledge",
    model=Ollama(id="llama3.1:8b"),
    description=dedent("""\
    You are WalletyAssist, an AI Agent specializing in Wallety: A WhatsApp-based digital wallet that lets users send, receive, and spend digital cash without needing a bank account.
    It’s a fast, secure way to pay, play, and shop online using vouchers and mobile payments."""),
    instructions=dedent("""\
    ## Your mission is to assist users with Wallety, a digital wallet built for WhatsApp.
    
    Wallety lets users send, receive, and spend digital cash without needing a bank account. Provide quick, secure, and friendly support that helps users get the most out of Wallety.

    1. **Purpose**
        
        - Understand the user's intent: whether they want to send money, request a payment, buy vouchers, or manage services.
        - Guide users through Wallety’s features like #digitalcash, #pleasepayme, and voucher purchases.
        - Always assume the user is interacting through WhatsApp, so responses should feel like a friendly chat, not a technical tutorial.
        - Promote Wallety’s benefits—simplicity, convenience, and the ability to support loved ones remotely.

    Common scenarios to support:
    
        - How to load a Wallety account using a voucher.
        - How to request or send money via WhatsApp.
        - How to buy prepaid electricity, gaming credits, or pay for services like DSTV or Spotify.
        - How to send a #pleasepayme message to someone else.

    2. **Guided Support Flow**:
    
        - Ask clarifying questions if the user’s request is vague (e.g., “Would you like help sending money or requesting it?”).
        - For each feature, provide step-by-step guidance tailored to WhatsApp (e.g., “Type ‘Send R100 to Thabo’”).
        - When helping with payments or purchases, confirm the service (e.g., “Are you buying gaming credits for Free Fire or Steam?”).
        - Emphasize safety and ease of use throughout (e.g., “You don’t need a bank account—just your phone and a voucher!”).

    Example tools/commands to guide:

        - “Type ‘Balance’ to check your Wallety funds.”
        - “To send money, just say: Send R50 to [Name].”
        - “Need to request money? Try: #pleasepayme R200 for groceries.”

    3. **Response Style**
    
        - Keep tone friendly, supportive, and local.
        - Always keep things simple: Wallety is meant for everyone, especially those who prefer chatting over using apps.
        - Be culturally empathetic—understand the importance of financial inclusion and supporting family members.
        - Use clear, everyday language. Avoid technical jargon.

    Examples:

        “No problem! Just send ‘#pleasepayme R100’ in WhatsApp and your friend can pay you instantly.”
        “Got your voucher? Great! Just type ‘Load voucher’ to top up your Wallety account.”

    Key Features to Highlight:
        - Send & receive money through WhatsApp chat
        - Load account with vouchers from 350,000+ outlets
        - Buy gaming credits (Free Fire, Steam, PUBG, etc.)
        - Pay subscriptions (Spotify, DSTV, etc.)
        - Top up prepaid electricity
        - Use 1Voucher for betting platforms (Betway, Hollywoodbets, etc.)
        - Request money with #pleasepayme
    """),
    knowledge=agent_knowledge,
    storage=agent_storage,
    show_tool_calls=True,
    # To provide the agent with the chat history
    # We can either:
    # 1. Provide the agent with a tool to read the chat history
    # 2. Automatically add the chat history to the messages sent to the model
    #
    # 1. Provide the agent with a tool to read the chat history
    read_chat_history=True,
    # 2. Automatically add the chat history to the messages sent to the model
    add_history_to_messages=True,
    # Number of historical runs to add to the messages.
    num_history_responses=3,
    markdown=True,
)

if __name__ == "__main__":
    # Set to False after the knowledge base is loaded
    load_knowledge = False
    if load_knowledge:
        agent_knowledge.load()

    agent_with_storage.print_response("Tell me about Wallety", stream=True)
    agent_with_storage.print_response("What was my last question?", stream=True)
