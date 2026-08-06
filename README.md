# Local RAG & Semantic Search Engine 

A local, privacy-first Retrieval-Augmented Generation (RAG) and Semantic Search Engine built in Python. This project was developed by deeply analyzing technical documentation and internalizing the RAG architecture to create a practical tool that streamlines personal research, document management, and information retrieval.

---

##  Why This Project? (The Engineering Journey)
Rather than blindly copying code, this project was built from the ground up by reading official documentation, understanding vector mathematics, and internalizing how RAG pipelines actually operate under the hood. 

The primary goal was to create a **real-world productivity tool** that simplifies daily workflows:
* **No More Manual Searching:** Instead of wasting hours scrolling through dense PDF documents or textbooks, I can instantly query my local library and pull out the exact context I need.
* **Privacy & Local Control:** Operating entirely offline ensures complete data privacy without relying on paid cloud services or external API rate limits.
* **Efficient Knowledge Management:** With smart chunking and incremental updates, the system acts as a personal, intelligent second brain for technical notes, articles, and references.

---

##  Tech Stack & Libraries
* **Python**
* **LangChain** (Document loading, text splitting, and vector store orchestration)
* **ChromaDB** (Persistent vector database)
* **Hugging Face Transformers** (`sentence-transformers/all-mpnet-base-v2` for dense embeddings)
* **Colorama** (Cross-platform colored terminal outputs)

---

## 📂 Project Structure
```text
LocalRAG/
│
├── data/                    # Directory containing your source PDF files
├── chroma_langchain_db/     # Persistent local Chroma vector database
├── main.py                  # Document loader, chunker, and vectorizer pipeline
├── query.py                 # Semantic similarity search script
└── README.md                # Project documentation
