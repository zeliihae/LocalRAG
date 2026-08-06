from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

YESIL = '\033[92m'
SARI = '\033[93m'
MAVI = '\033[94m'
KIRMIZI = '\033[91m'
SIFIRLA = '\033[0m'

def load_documents():
    document_loader = PyPDFDirectoryLoader(r"C:\Users\DELL\Desktop\LocalRAG\data")
    return document_loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,  
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
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
        
    return chunks

# --- KODUN ANA AKIŞI ---
documents = load_documents()
chunks = split_documents(documents)
chunks = calculate_chunk_ids(chunks)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# SADECE BU FONKSİYON GÜNCELLENDİ
def add_to_chroma(chunks):
    db = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db" 
    )
    
    # 1. Veritabanındaki mevcut ID'leri çek
    existing_items = db.get(include=[])  
    existing_ids = set(existing_items["ids"])
    print(f"Veritabanındaki mevcut parça sayısı: {len(existing_ids)}")

    # 2. Sadece veritabanında OLMAYAN yeni parçaları ayır
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # 3. Yeni parça varsa kaydet, yoksa uyarı ver
    if len(new_chunks) > 0:
        print(f"{MAVI}Yeni eklenen parça sayısı: {len(new_chunks)}{SIFIRLA}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(documents=new_chunks, ids=new_chunk_ids)
        print(F"{YESIL}Yeni dokümanlar ChromaDB'ye başarıyla kaydedildi!{SIFIRLA}")
    else:
        print(F"{KIRMIZI}Eklenecek yeni doküman bulunamadı. Veritabanı zaten güncel.{SIFIRLA}")

# Fonksiyonu çalıştır
add_to_chroma(chunks)