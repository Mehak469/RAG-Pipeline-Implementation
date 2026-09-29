from langchain_text_splitters import RecursiveCharacterTextSplitter

#divide this whole text into chunks->allowed chunk size=10


text="""
My name is Mehak Gul.
I'm 19 years old.

I live in sgd.
What's happening?
"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=0
)


results=splitter.split_text(text)


print(results)
print(len(results))