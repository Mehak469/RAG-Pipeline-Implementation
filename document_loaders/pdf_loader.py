from langchain_community.document_loaders import PyPDFLoader


loader= PyPDFLoader(f"./data/CA.pdf")


docs=loader.load()

print(len(docs))
print(docs[0])