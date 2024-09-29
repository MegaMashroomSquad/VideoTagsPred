from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

llm = Ollama(model="llama3", temperature=0)

def process_title_with_llama(title, prompt_template):
    prompt = prompt_template.format(title=title)
    response = llm.invoke(prompt)
    return response