from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq LLM
model = ChatGroq(
    model="llama-3.1-8b-instant",   # or "mixtral-8x7b-32768"
    temperature=0.3
)

# Prompt template
prompt = PromptTemplate(
    template="Write a summary for the following poem:\n{poem}",
    input_variables=["poem"]
)

# Output parser
parser = StrOutputParser()

# Load text file
loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

# Debug prints
print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

# Create chain
chain = prompt | model | parser

# Invoke chain
result = chain.invoke({"poem": docs[0].page_content})
print(result)
