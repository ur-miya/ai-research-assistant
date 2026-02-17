import chromadb
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100

def init_chromadb():
    # Инициализация ChromaDB
    client = chromadb.PersistentClient(path="./chromadb_data")

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Удалена существующая коллекция")
    except:
        pass
    
    # Новая коллекция
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} 
    )
    
    print(f"Создана коллекция {COLLECTION_NAME}")
    return client, collection

def load_documents():
    # Загрузка документов
    with open("processed_docs.json", "r", encoding='utf-8') as f:
        docs = json.load(f)
    print(f"Загружено {len(docs)} документов")
    return docs

def index_documents(collection, docs):
    # Индексация документов в ChromaDB
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Загружена модель эмбеддингов: {EMBEDDING_MODEL}")
    
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Индексация батчами"):
        batch = docs[i:i+BATCH_SIZE]
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for j, doc in enumerate(batch):
            # ID для ChromaDB
            ids.append(f"doc_{i+j}")
            # Эмбеддинг
            embedding = encoder.encode(doc["text"]).tolist()
            embeddings.append(embedding)
            # Метаданные
            metadatas.append({
                "title": doc["metadata"]["title"][:100],  
                "authors": str(doc["metadata"]["authors"])[:100],
                "categories": doc["metadata"]["categories"][:50],
                "published": doc["metadata"]["published"]
            })
            # Часть текста для отображения
            documents.append(doc["text"][:500]) 
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
    
    print(f"Проиндексировано: {len(docs)} документов")

def test_search(collection):
    # Тестирование поиска
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    test_queries = [
        "machine learning neural networks",
        "quantum physics",
        "mathematics optimization"
    ]
    
    print("\nТестирование поиска:")
    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        query_vector = encoder.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        for i in range(len(results['ids'][0])):
            print(f"  Релевантность: {results['distances'][0][i]:.3f}")
            print(f"   {results['metadatas'][0][i]['title']}...")
            print()

def simple_search_interface(collection):
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    print("Введите 'quit' для выхода\n")
    
    while True:
        query = input("\nЗапрос: ")
        if query.lower() == 'quit':
            break
        if not query.strip():
            continue
        query_vector = encoder.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5
        )
        
        print(f"\nНайдено результатов: {len(results['ids'][0])}")
        for i in range(len(results['ids'][0])):
            print(f"{i+1}. [Релевантность: {1 - results['distances'][0][i]:.3f}]")
            print(f"    {results['metadatas'][0][i]['title']}")
            print(f"    Категории: {results['metadatas'][0][i]['categories']}")
            print(f"    {results['metadatas'][0][i]['published']}")

if __name__ == "__main__":
    client, collection = init_chromadb()
    docs = load_documents()
    index_documents(collection, docs)
    test_search(collection)
    simple_search_interface(collection)
    print("\n Данные сохранены в папке ./chromadb_data")