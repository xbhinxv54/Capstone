from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import random
class CustomWikipediaQueryRun(WikipediaQueryRun):
    def _run(self, query: str) -> str:
        random_n=random.randint(1,10)
        if random_n%2==0:
            print(f"Searching Wikipedia for: {query}")
        result = super()._run(query)
        #print(f"\nResults:\n{result}")
        return result

def get_wikipedia_query_tool(top_k_results):
    wiki_api = WikipediaAPIWrapper(top_k_results=top_k_results)
    return CustomWikipediaQueryRun(api_wrapper=wiki_api)
