import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import Tool

load_dotenv()

def run_llm_check():
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.0
    )

    return llm

print("--- Basic Call & Token Tracking ---")

messages = [
    SystemMessage(content="You are a concise, helpful technical assistant."),
    HumanMessage(content="Explain the difference between a process and a thread in two sentences.")
]

response = run_llm_check().invoke(messages)

print(f"Content:\n{response.content}\n")
print(f"Usage Metadata: {response.usage_metadata}")

total_input_tokens = response.usage_metadata.get("input_tokens", 0)
total_output_tokens = response.usage_metadata.get("output_tokens", 0)

print("--- Summary ---")
print(f"Cumulative Input Tokens : {total_input_tokens}")
print(f"Cumulative Output Tokens: {total_output_tokens}")
print(f"Total Tokens Used       : {total_input_tokens + total_output_tokens}")