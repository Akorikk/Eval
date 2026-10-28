
from models import model
from tools import (
    calculate_right_refund,
    get_today_date,
    get_user_purchase_history,
    issue_refund_ticket,
)
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from prompt import agent_system_prompt
from objects import Answer
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from uuid import uuid4

load_dotenv()

agent = create_agent(
    model=model,
    system_prompt=agent_system_prompt,
    tools=[
        calculate_right_refund,
        get_user_purchase_history,
        issue_refund_ticket,
        get_today_date,
    ],
    response_format=ToolStrategy(Answer),
    checkpointer=InMemorySaver(),
)


def run_conversation():
    config = {"configurable": {"thread_id": str(uuid4())}}

    question = HumanMessage(
        content="I would like to request a refund"
    )
    conversation = True

    while conversation:
        response = agent.invoke(
            {"messages": [question]},
            config=config,
        )

        structured_response = response["structured_response"]

        print(structured_response.answer)
        conversation = structured_response.conversation

        if conversation:
            user_input = input("Reply:\n")
            question = HumanMessage(content=user_input)

    print("The job is done!")
