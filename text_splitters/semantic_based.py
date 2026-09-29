from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv,find_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv(find_dotenv())

sample_text = """
Photosynthesis is the process by which green plants convert light energy into chemical energy. It takes place mainly in the chloroplasts, which contain the pigment chlorophyll. Chlorophyll absorbs sunlight, especially in the red and blue wavelengths. The plant uses this energy to combine carbon dioxide from the air with water from the soil. The result is glucose, which fuels growth, and oxygen, which is released into the atmosphere. Without photosynthesis, most life on Earth could not exist.

The French Revolution began in 1789 and transformed the political landscape of Europe. Financial crisis and food shortages had left the French population deeply unhappy with the monarchy. The storming of the Bastille on July 14th became a powerful symbol of popular uprising. The revolutionaries later adopted the Declaration of the Rights of Man and of the Citizen. King Louis XVI was eventually executed, and the country entered a turbulent period known as the Reign of Terror. The revolution ended when Napoleon Bonaparte seized power in 1799.

Python is a high-level programming language known for its readable syntax. It supports multiple paradigms, including object-oriented, functional, and procedural programming. Developers use Python for web development, data analysis, automation, and machine learning. Libraries such as NumPy, pandas, and PyTorch have made it the dominant language in data science. Its large community provides thousands of open-source packages through the Python Package Index. Beginners often choose Python as their first language because it is easy to learn.

Making a good espresso requires attention to grind size, water temperature, and pressure. The coffee beans should be ground fine, but not so fine that the water cannot pass through. Water at around ninety-three degrees Celsius extracts the best flavors from the grounds. A standard shot is pulled in about twenty-five to thirty seconds using nine bars of pressure. If the shot runs too quickly, it will taste sour and weak. If it runs too slowly, it will taste bitter and harsh.
"""

splitter=SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.6,
)

chunks=splitter.split_text(sample_text)

print(len(chunks))
print(chunks)

print()

