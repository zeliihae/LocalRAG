from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from colorama import Fore, init


init(autoreset=True)

def load_documents():
    print(Fore.CYAN + "Loading documents from PDF directory...")
    document_loader = PyPDFDirectoryLoader(r"C:\Users\DELL\Desktop\LocalRAG\data")
    documents = document_loader.load()
    print(Fore.GREEN + f"Loaded {len(documents)} document pages successfully.")
    return documents

def split_documents(documents):
    print(Fore.CYAN + "Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,  
    )
    chunks = text_splitter.split_documents(documents)
    print(Fore.GREEN + f"Created {len(chunks)} text chunks in total.")
    return chunks

def calculate_chunk_ids(chunks):
    print(Fore.CYAN + "Calculating unique chunk IDs...")
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        last_page_id = current_page_id
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        
    print(Fore.GREEN + "Chunk IDs calculated successfully.")
    return chunks

print(Fore.YELLOW + "Initializing HuggingFace Embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

def add_to_chroma(chunks):
    print(Fore.CYAN + "Connecting to ChromaDB...")
    db = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db" 
    )
    
    # 1. Veritabanındaki mevcut ID'leri çek
    existing_items = db.get(include=[])  
    existing_ids = set(existing_items["ids"])
    print(Fore.YELLOW + f"Current number of chunks in database: {len(existing_ids)}")

    # 2. Sadece veritabanında OLMAYAN yeni parçaları ayır
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # 3. Yeni parça varsa kaydet, yoksa uyarı ver
    if len(new_chunks) > 0:
        print(Fore.CYAN + f"Number of new chunks to add: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(documents=new_chunks, ids=new_chunk_ids)
        print(Fore.GREEN + "New documents successfully saved to ChromaDB!")
    else:
        print(Fore.RED + "No new documents found. Database is already up to date.")

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    chunks = calculate_chunk_ids(chunks)
    
   
    add_to_chroma(chunks)