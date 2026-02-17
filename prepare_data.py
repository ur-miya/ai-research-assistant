import json
import uuid
from tqdm import tqdm

def load_articles(file_path, max_records=5000):
    articles = []
    print(f"Загрузка статей из {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Чтение файла")):
            if i >= max_records:
                break
            try:
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1]
                # Преобразование JSON строку в словарь
                article = json.loads(line)
                articles.append(article)
                
            except json.JSONDecodeError as e:
                print(f"Ошибка в строке {i}: {e}")
                continue
    print(f"Загружено {len(articles)} статей")
    return articles

def create_documents(articles):
    # Создание документов для индексации из списка статей
    docs = []
    
    for i, article in enumerate(tqdm(articles, desc="Обработка статей")):
        doc_id = article.get("id", str(uuid.uuid4()))
        title = article.get("title", "Без заголовка")
        abstract = article.get("abstract", "Нет аннотации")
        text = f"Title: {title}\n\nAbstract: {abstract}"
        
        metadata = {
            "title": title[:200] + "..." if len(title) > 200 else title,  
            "authors": article.get("authors", "Неизвестен"),
            "categories": article.get("categories", ""),
            "published": article.get("update_date", "Неизвестно"),
            "id": doc_id
        }
        
        docs.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata
        })
    
    return docs

def save_documents(docs, output_file):
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"Сохранено {len(docs)} документов в {output_file}")

def print_sample(docs, n=1):
    if docs:
        print("\nПример обработанного документа:")
        print(f"ID: {docs[0]['id']}")
        print(f"Заголовок: {docs[0]['metadata']['title']}")
        print(f"Авторы: {docs[0]['metadata']['authors']}")
        print(f"Категории: {docs[0]['metadata']['categories']}")
        print(f"Текст для поиска (первые 200 символов):")
        print(f"{docs[0]['text'][:200]}...")

if __name__ == "__main__":
    articles = load_articles("arxiv_sample.json", max_records=5000)
    if not articles:
        print("Нет статей для обработки")
        exit()
    # Создание документов для индексации
    docs = create_documents(articles)
    save_documents(docs, "processed_docs.json")
    # Пример документа
    print_sample(docs)
