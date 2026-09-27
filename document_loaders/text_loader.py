from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# LLM
model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash"
)

# Prompt
prompt = ChatPromptTemplate.from_template(
    "what about this document is :\n\n{doc}"
)

# Output parser
parser = StrOutputParser()


# Load document
loader = TextLoader(
    f"./data/post.txt",
    encoding="utf-8"
)

docs = loader.load()

print(docs)
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)


# Chain
chain = prompt | model | parser

response = chain.invoke({
    "doc": docs[0].page_content
})

print(response)