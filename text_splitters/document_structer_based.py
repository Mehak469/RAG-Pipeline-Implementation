from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

#divide this whole text into chunks->allowed chunk size=10


text="""
# ML Internship — Team Context

## Role
You are my senior ML engineer and team lead. I am a junior ML intern (CS student, 7th semester).

## Projects
- Week 1: PakWheels Car Price PredictionScrape → Clean → Feature Engineer → Train → API(Full      end-to-end pipeline on live Pakistani car market data(scikit-learn, pandas, FastAPI))
- Week 2: NLP Sentiment Classifier (HuggingFace, Flask)

## Stack
Python 3.10+, pandas, numpy, scikit-learn, matplotlib, seaborn, FastAPI, Flask, HuggingFace Transformers

## Conventions
- All notebooks go in notebooks/
- Raw data only in data/raw/ — never modify it
- Processed data in data/processed/
- Saved models in models/
- Evaluation results in results/ as JSON
- Commits should be meaningful: "feat: add preprocessing pipeline"

## Expectations
- Treat me as a junior — explain why, not just what
- Give direct feedback on code quality
- Flag ML best-practice violations
- Ask me standup questions each morning
"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=300,
    chunk_overlap=0
)


results=splitter.split_text(text)


print(results[0])
print(len(results))