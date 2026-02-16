import json
import uuid
from tqdm import tqdm

def load_articles(file_path, max_records=5000):
    """
    Загружает статьи из arXiv JSON файла.
    
    Аргументы:
        file_path: путь к файлу arxiv_sample.json
        max_records: сколько статей обработать (5000 достаточно)
    
    Возвращает:
        список статей
    """
    articles = []
    print(f"Загрузка статей из {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Чтение файла")):
            if i >= max_records:
                break
            try:
                # Удаляем запятую в конце строки, если есть
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1]
                
                # Преобразуем JSON строку в словарь
                article = json.loads(line)
                articles.append(article)
                
            except json.JSONDecodeError as e:
                print(f"Ошибка в строке {i}: {e}")
                continue
    
    print(f"Загружено {len(articles)} статей")
    return articles

def create_documents(articles):
    """
    Создаёт документы для индексации из списка статей.
    
    Каждый документ содержит:
    - id: уникальный идентификатор
    - text: объединённый заголовок и аннотация (для поиска)
    - metadata: метаданные для фильтрации
    """
    docs = []
    
    for i, article in enumerate(tqdm(articles, desc="Обработка статей")):
        # Получаем ID статьи или создаём новый
        doc_id = article.get("id", str(uuid.uuid4()))
        
        # Получаем заголовок и аннотацию
        title = article.get("title", "Без заголовка")
        abstract = article.get("abstract", "Нет аннотации")
        
        # Объединяем для поиска
        text = f"Title: {title}\n\nAbstract: {abstract}"
        
        # Подготавливаем метаданные
        metadata = {
            "title": title[:200] + "..." if len(title) > 200 else title,  # обрезаем длинные названия
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
    """
    Сохраняет обработанные документы в JSON файл.
    """
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"Сохранено {len(docs)} документов в {output_file}")

def print_sample(docs, n=1):
    """
    Печатает пример обработанного документа для проверки.
    """
    if docs:
        print("\nПример обработанного документа:")
        print(f"ID: {docs[0]['id']}")
        print(f"Заголовок: {docs[0]['metadata']['title']}")
        print(f"Авторы: {docs[0]['metadata']['authors']}")
        print(f"Категории: {docs[0]['metadata']['categories']}")
        print(f"Текст для поиска (первые 200 символов):")
        print(f"{docs[0]['text'][:200]}...")

if __name__ == "__main__":
    # 1. Загружаем статьи
    articles = load_articles("arxiv_sample.json", max_records=5000)
    
    if not articles:
        print("Нет статей для обработки")
        exit()
    
    # 2. Создаём документы
    print("\nСоздание документов для индексации...")
    docs = create_documents(articles)
    
    # 3. Сохраняем результат
    save_documents(docs, "processed_docs.json")
    
    # 4. Показываем пример
    print_sample(docs)
    
    print("\nМожно индексировать документы в Qdrant.")