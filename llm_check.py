import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

def run_llm_check():
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.0
    )
    
    response = llm.invoke([
        HumanMessage(content="Say the words 'Inference engine is online' and nothing else.")
    ])
    
    print(response.content.strip())

if __name__ == "__main__":
    run_llm_check()