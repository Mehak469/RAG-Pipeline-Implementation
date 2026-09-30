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

data=vector_store.get(include=["embeddings","documents","metadatas"])

print(data)

sim=vector_store.similarity_search(
    query="which doc is about programming?",
    k=1   #depends how many similiar you want...
)

print(sim)


sim_score=vector_store.similarity_search_with_score(
    query="which doc is about programming?",
    k=3   #depends how many similiar you want & score represents ditance among vectors
)

print(sim_score)


filter_metadata=vector_store.get(
    where={
        "year":2023
    },
    include=["documents"]
)

print(filter_metadata)


updated_doc=Document(
    page_content="I'm learning RAG through Langchain in Python",
    metadata={"topic": "programming", "year": 2023}
)

vector_store.update_document(document_id="242893be-1bff-47b4-889c-a9a8983c49b9",document=updated_doc)

print(data)


vector_store.delete(
    ids=["242893be-1bff-47b4-889c-a9a8983c49b9"]
)

print(data)

