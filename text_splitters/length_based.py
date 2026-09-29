# from langchain_text_splitters import CharacterTextSplitter

# text="""Post#01

# A concept that changed the future of tech: The Transformer

# In 2017, a group of researchers at Google published a paper that laid the foundation for the current AI era: "Attention Is All You Need."

# Most beginners feel overwhelmed trying to understand it — but the core idea is simpler than it looks.

# At its heart, a Transformer just predicts the next word. Here's how:

# → Embeddings: turn words into lists of numbers (so the model can do math on language)
# → Attention: figure out which words are relevant to which other words
# → Neural network: process all of that into a set of possible next words
# → Output: pick the word with the highest probability

# That's it. Everything from ChatGPT to Claude is built on this loop, repeated at massive scale.

# What part of this trips you up the most — embeddings or attention?"""

# print(len(text))


# splitter=CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=0,
#     separator=""
# )

# result=splitter.split_text(text)

# print(result)
# print(type(result))
# print(len(result))


#worflow from dcument_loaders --> text_splitters

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


loader=PyPDFLoader(f"D://UPSkilling//RAG//document_loaders//data//FYP.pdf")

docs=loader.load()

print(len(docs))

splitter=CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=5,
    separator=""
)


results=splitter.split_documents(docs)

print(len(results[0].page_content))
print(results[0].page_content)

print(len(results))
print(results)