from langchain_community.document_loaders import WebBaseLoader
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
    "Write an answer the following qst\n {question} from the fllowing text \n\n{doc}"
)

# Output parser
parser = StrOutputParser()


url="https://www.pakwheels.com/used-cars/search/-/"

loader=WebBaseLoader(url)

docs=loader.load()

print(len(docs))  #only one doc for single url(can be sent list of urls)
print(docs[0].page_content)


# Chain
chain = prompt | model | parser

response = chain.invoke({
    "question":"What is the price of Toyota Vitz and also all the spec detaila with it",
    "doc": docs[0].page_content
})

print(response)