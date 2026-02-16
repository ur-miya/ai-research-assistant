import chromadb
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

# Конфигурация
COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100

def init_chromadb():
    """Инициализация ChromaDB (локально)"""
    # Создаём клиент (данные сохранятся в папке ./chromadb_data)
    client = chromadb.PersistentClient(path="./chromadb_data")
    
    # Удаляем старую коллекцию если есть
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Удалена существующая коллекция")
    except:
        pass
    
    # Создаём новую коллекцию
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # используем косинусное расстояние
    )
    
    print(f"Создана коллекция {COLLECTION_NAME}")
    return client, collection

def load_documents():
    """Загрузка обработанных документов"""
    with open("processed_docs.json", "r", encoding='utf-8') as f:
        docs = json.load(f)
    print(f"Загружено {len(docs)} документов")
    return docs

def index_documents(collection, docs):
    """Индексация документов в ChromaDB"""
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Загружена модель эмбеддингов: {EMBEDDING_MODEL}")
    
    # Подготавливаем данные для батчевой загрузки
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Индексация батчами"):
        batch = docs[i:i+BATCH_SIZE]
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for j, doc in enumerate(batch):
            # ID для ChromaDB (должен быть строкой)
            ids.append(f"doc_{i+j}")
            
            # Создаём эмбеддинг
            embedding = encoder.encode(doc["text"]).tolist()
            embeddings.append(embedding)
            
            # Метаданные
            metadatas.append({
                "title": doc["metadata"]["title"][:100],  # обрезаем длинные названия
                "authors": str(doc["metadata"]["authors"])[:100],
                "categories": doc["metadata"]["categories"][:50],
                "published": doc["metadata"]["published"]
            })
            
            # Текст для отображения
            documents.append(doc["text"][:500])  # храним только часть текста
        
        # Добавляем в ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        print(f"⬆Загружено {min(i+BATCH_SIZE, len(docs))} документов")
    
    print(f"Индексация завершена! Всего: {len(docs)} документов")

def test_search(collection):
    """Тестирование поиска"""
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
    """Простой интерфейс поиска"""
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    print("\n" + "="*50)
    print("ПОИСК ПО НАУЧНЫМ СТАТЬЯМ")
    print("="*50)
    print("Введите 'quit' для выхода\n")
    
    while True:
        query = input("\nВаш запрос: ")
        if query.lower() == 'quit':
            break
        
        if not query.strip():
            continue
        
        # Создаём эмбеддинг запроса
        query_vector = encoder.encode(query).tolist()
        
        # Ищем похожие
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5
        )
        
        print(f"\nНайдено результатов: {len(results['ids'][0])}")
        print("-" * 50)
        
        for i in range(len(results['ids'][0])):
            print(f"{i+1}. [Релевантность: {1 - results['distances'][0][i]:.3f}]")
            print(f"    {results['metadatas'][0][i]['title']}")
            print(f"    Категории: {results['metadatas'][0][i]['categories']}")
            print(f"    {results['metadatas'][0][i]['published']}")
            
            # Показываем фрагмент текста
            text = results['documents'][0][i]
            print(f"    {text[:200]}...")
            print()

if __name__ == "__main__":
    print(" Начало работы с ChromaDB...")
    
    # Инициализация
    client, collection = init_chromadb()
    
    # Загрузка документов
    docs = load_documents()
    
    # Индексация
    index_documents(collection, docs)
    
    # Тестирование
    test_search(collection)
    
    # Запускаем поисковый интерфейс
    simple_search_interface(collection)
    
    print("\n Данные сохранены в папке ./chromadb_data")