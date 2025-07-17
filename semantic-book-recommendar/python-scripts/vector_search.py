from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from tabulate import tabulate
import pandas as pd

#loading the .env file
load_dotenv()

books = pd.read_csv("books_cleaned.csv")

books["tagged_description"].to_csv("tagged_description.txt",
									sep = "\n",
									index = False,
									header = False)

"""Our existing Chroma DB (chroma_db_books) was created with OpenAIEmbeddings, which produce 1536-dimensional vectors.
On the other hand, HuggingFaceEmbeddings produces 384-dimensional vectors.
"""

# OpenAI approach using its API
# load the documents and instantiate the text-splitter
# the chunk size it set to zero to prioritize splitting at the separator rather than the chunk-size, hence we might warnings
"""raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

# checking if it prints the first description correctly
print(documents[0])

#create the document embeddings and store them in the vector database
db_books = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="chroma_db_books"
)
print("Vector database stored to local disk:)")
"""

# HuggingFace approach >> to save money
# conditional flag to avoid creating vector database everytime
query = "A book to teach children about nature"

REBUILD_VECTOR_DB = False
PERSIST_DIR = "chroma_db_books_hf"
MODEL = "sentence-transformers/all-MiniLM-L6-v2" #384-dim (keep consistent!)

# 1. Define the embedding model (same for build & query)
embedding = HuggingFaceEmbeddings(model_name=MODEL)

if REBUILD_VECTOR_DB:
    # 2. Load and split text
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)

    # 3. Create and persist vector DB
    db_books = Chroma.from_documents(
        documents,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )

    print("First split chunk:")
    print(documents[0].page_content)

else:
    # 4. Load existing DB (no re-embedding)
    db_books = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding
    )

    # 5. Run a query
    results = db_books.similarity_search(query, k=1)
    print("Top semantic match:\n" + results[0].page_content + "\n")

docs = db_books.similarity_search(query, k = 10)
print("First 10 results: \n", docs, "\n")

# filters and gives the isbn for the first result from the query results
print("First result of all:\n")
print(books[books["isbn13"] == int(docs[0].page_content.split()[0].strip())])
print("\n")

def retrieve_semantic_recommendations(
		query: str,
		top_k: int = 10,
) -> pd. DataFrame:
	recs = db_books.similarity_search (query, k = 50)

	books_list = []

	for i in range(0, len(recs)):
		books_list += [int(recs[i].page_content.strip('"').split()[0])]

	return books[books["isbn13"].isin(books_list)].head(top_k)

results = retrieve_semantic_recommendations(query)
print("Recommendations:\n")
print(tabulate(results, headers='keys', tablefmt='grid', showindex=False))











