from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.thinking import ThinkingTools
from agno.tools.yfinance import YFinanceTools

thinking_agent = Agent(
    model=Ollama(id="llama3.1:8b"),
    tools=[
        ThinkingTools(add_instructions=True),
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        ),
    ],
    instructions="Use tables where possible",
    show_tool_calls=True,
    markdown=True,
)
thinking_agent.print_response("Write a report comparing NVDA to TSLA", stream=True)
