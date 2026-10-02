from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

embeddigs=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


docs = [
    Document(
        page_content="Python is a popular programming language known for its simple syntax and wide use in data science and machine learning.",
        metadata={"topic": "programming", "year": 2023},
    ),
    Document(
        page_content="The Eiffel Tower in Paris was completed in 1889 and attracts millions of tourists every year.",
        metadata={"topic": "travel", "year": 2021},
    ),
    Document(
        page_content="Neural networks are inspired by the human brain and learn patterns from large amounts of data.",
        metadata={"topic": "ai", "year": 2024},
    ),
    Document(
        page_content="A balanced diet with vegetables, fruits, and enough water helps maintain good health and energy.",
        metadata={"topic": "health", "year": 2022},
    ),
    Document(
        page_content="Cricket is one of the most-watched sports in South Asia, with fans following matches passionately.",
        metadata={"topic": "sports", "year": 2023},
    ),
]

vector_store=Chroma(
    embedding_function=embeddigs,
    persist_directory="chorma_db",
    collection_name="example_collecion"
)

vector_store.add_documents(docs)

# Same as Vectr stores ability(vanila retriver)

retriever=vector_store.as_retriever(search_kwargs={"k":2})

query="What's aboout year 2023?"

results=retriever.invoke(query)


for i,doc in enumerate(results):
    print(f"\n ==== Results{i+1} === ")
    print(f"Content \n {doc.page_content}....")