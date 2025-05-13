from dotenv import load_dotenv
import os

from agno.agent import Agent
from agno.cli.console import console
from agno.models.deepseek import DeepSeek
from agno.models.ollama import Ollama

# Load environment variables from .env
load_dotenv()

task = "9.11 and 9.9 -- which is bigger?"

regular_agent_claude = Agent(model=Ollama(id="deepseek-r1:8b"))
reasoning_agent_claude = Agent(
    model=Ollama(id="deepseek-r1:8b"),
    reasoning_model=DeepSeek(id="deepseek-reasoner"),
)

regular_agent_openai = Agent(model=Ollama(id="llama3.1:8b"))
reasoning_agent_openai = Agent(
    model=Ollama(id="llama3.1:8b"),
    reasoning_model=DeepSeek(id="deepseek-reasoner"),
)

console.rule("[bold blue]Regular Claude Agent[/bold blue]")
regular_agent_claude.print_response(task, stream=True)

console.rule("[bold green]Claude Reasoning Agent[/bold green]")
reasoning_agent_claude.print_response(task, stream=True)

console.rule("[bold red]Regular OpenAI Agent[/bold red]")
regular_agent_openai.print_response(task, stream=True)

console.rule("[bold yellow]OpenAI Reasoning Agent[/bold yellow]")
reasoning_agent_openai.print_response(task, stream=True)
