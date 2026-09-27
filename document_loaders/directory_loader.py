from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader


loader= DirectoryLoader(
    path='data',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs = loader.load()

# for document in docs:
#     print(document.metadata)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)

print(len(docs))
print(docs[2])