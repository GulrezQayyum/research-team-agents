from langchain_community.tools import DuckDuckGoSearchResults

def run_search_check():
    search_tool = DuckDuckGoSearchResults(max_results=3)
    
    results = search_tool.invoke({"query": "What is the capital of France?"})
    
    print("--- Search Results Summary ---")
    print(results)

if __name__ == "__main__":
    run_search_check()