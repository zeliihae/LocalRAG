from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from colorama import Fore, Style, init


init(autoreset=True)

CHROMA_PATH = "chroma_langchain_db"

def search_database(query_text: str):
    print(Fore.CYAN + "Connecting to the database and initiating vector search...")
    

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )

    results = db.similarity_search_with_score(query_text, k=3)

    if len(results) == 0:
        print(Fore.RED + "No matching information found.")
        return

    print(Fore.GREEN + f"\n🔍 Most relevant PDF chunks found for '{query_text}':\n")

    for i, (doc, score) in enumerate(results, start=1):
        print(Fore.YELLOW + f"--- Result {i} (Vector Distance: {score:.4f}) ---")
        print(Style.BRIGHT + Fore.WHITE + doc.page_content)
        print(Fore.MAGENTA + f"📂 Source Document: {doc.metadata.get('id', 'Unknown')}\n")

if __name__ == "__main__":
    test_query = "What is artificial intelligence?" 
    search_database(test_query)