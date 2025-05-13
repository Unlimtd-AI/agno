from dotenv import load_dotenv
import os

from agno.agent import Agent
# from agno.models.deepseek import DeepSeek
from agno.models.ollama import Ollama

# Load environment variables from .env
# load_dotenv()

task = (
    "You are a train conductor faced with an emergency: the brakes have failed, and the train is heading towards "
    "five people tied on the track. You can divert the train onto another track, but there is one person tied there. "
    "Do you divert the train, sacrificing one to save five? Provide a well-reasoned answer considering utilitarian "
    "and deontological ethical frameworks. "
    "Provide your answer also as an ascii art diagram."
)

reasoning_agent = Agent(
    model=Ollama(id="deepseek-r1:8b"),
    reasoning_model=Ollama(id="wizardlm2:7b"),
    markdown=True,
     reasoning_preamble="Please think step by step before answering. Use both utilitarian and deontological logic.",
)
reasoning_agent.print_response(task, stream=True)
