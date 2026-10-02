from langchain_community.retrievers import WikipediaRetriever


retriever=WikipediaRetriever(
    top_k=2,
    lang="en"
)

query="What's the most recent update from OpenAI?"


docs=retriever.invoke(query)

print(docs)
print(len(docs))
print(docs[0].page_content)


for i,doc in enumerate(docs):
    print(f"\n ==== Results{i+1} === ")
    print(f"Content \n {doc.page_content}....")