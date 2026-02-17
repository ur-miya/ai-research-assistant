import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
import json

from langgraph.graph import StateGraph, END

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

TARGET_CATEGORIES = {
    # Machine Learning
    'cs.LG': 'Machine Learning',
    'cs.AI': 'Artificial Intelligence',
    'cs.CL': 'Computation and Language',
    'cs.CV': 'Computer Vision',
    'cs.NE': 'Neural and Evolutionary Computing',
    # Data Science
    'cs.DB': 'Databases',
    'cs.IR': 'Information Retrieval',
    'stat.ML': 'Statistics - Machine Learning',
    # Mathematics relevant to ML
    'math.OC': 'Optimization and Control',
    'math.ST': 'Statistics Theory',
    'stat.TH': 'Statistics Theory'
}

PRIORITY_CATEGORIES = ['cs.LG', 'cs.AI', 'cs.CL', 'cs.CV', 'cs.NE', 'cs.DB', 'cs.IR', 'stat.ML']

def is_relevant_category(categories_str: str) -> bool:
    # Проверка, относится ли статья к интересующим нас категориям.
    if not categories_str:
        return False
    
    categories = categories_str.split()
    
    for cat in categories:
        if cat in TARGET_CATEGORIES:
            return True
        for target in TARGET_CATEGORIES:
            if cat.startswith(target):
                return True
    
    return False

def get_category_priority(categories_str: str) -> int:
    if not categories_str:
        return 0
    
    categories = categories_str.split()
    
    for i, priority_cat in enumerate(PRIORITY_CATEGORIES):
        for cat in categories:
            if cat.startswith(priority_cat):
                return len(PRIORITY_CATEGORIES) - i 
    
    return 1 

class AgentState(TypedDict):
    messages: List[Any]
    query: str
    original_query: str
    search_results: List[Dict]
    filtered_results: List[Dict]
    analysis: str
    final_answer: str
    steps: List[str]
    filter_stats: Dict[str, int]

class ResearchTools:
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chromadb_data")
        self.collection = self.client.get_collection("arxiv_papers")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def search_papers(self, query: str, n_results: int = 20) -> str:
        try:
            query_vector = self.encoder.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )
            
            papers = []
            for i in range(len(results['ids'][0])):
                relevance = 1 - results['distances'][0][i]
                papers.append({
                    'relevance': round(relevance, 3),
                    'title': results['metadatas'][0][i]['title'],
                    'categories': results['metadatas'][0][i]['categories'],
                    'published': results['metadatas'][0][i]['published'],
                    'abstract': results['documents'][0][i][:500],
                    'id': results['ids'][0][i]
                })
            
            return json.dumps(papers, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def analyze_results(self, query: str, papers: List[Dict]) -> str:
        if not papers:
            return "По запросу ничего не найдено"
        
        analysis = f"Анализ результатов по запросу: '{query}'\n\n"
        analysis += f"Найдено статей: {len(papers)}\n\n"
        
        categories = {}
        for paper in papers:
            cats = paper['categories'].split()
            for cat in cats:
                if cat in TARGET_CATEGORIES:
                    cat_name = TARGET_CATEGORIES[cat]
                    categories[cat_name] = categories.get(cat_name, 0) + 1
                elif cat.startswith('cs.'):
                    categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            analysis += "Распределение по категориям:\n"
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                analysis += f"  - {cat}: {count} статей\n"

        analysis += "\nНаиболее релевантные статьи\n"
        for i, paper in enumerate(papers[:3], 1):
            analysis += f"\n{i}. {paper['title']}\n"
            analysis += f"   - Релевантность: {paper['relevance']}\n"
            analysis += f"   - Категории: {paper['categories']}\n"
            
            cat_descriptions = []
            for cat in paper['categories'].split():
                if cat in TARGET_CATEGORIES:
                    cat_descriptions.append(TARGET_CATEGORIES[cat])
            if cat_descriptions:
                analysis += f"   - Область: {', '.join(cat_descriptions[:2])}\n"
            
            analysis += f"   - Аннотация: {paper['abstract'][:200]}...\n"
        
        return analysis

tools = ResearchTools()

def node_plan_research(state: AgentState) -> AgentState:
    original_query = state["original_query"]
    enhanced_query = original_query
    if any(c in original_query for c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
        keywords = {
            'нейронн': 'neural network',
            'сет': 'network',
            'обучен': 'learning',
            'искусственн': 'artificial intelligence',
            'обработк': 'natural language processing',
            'дистилляц': 'knowledge distillation',
            'язык': 'language',
            'текст': 'text',
        }

        added = []
        for ru_word, en_term in keywords.items():
            if ru_word in original_query.lower():
                enhanced_query += f" {en_term}"
                added.append(en_term)
        
        #if added:
            #state["steps"] = state.get("steps", []) + [f"Добавлены английские термины: {', '.join(added)}"]
    
    steps = state.get("steps", [])
    #steps.append(f"Планирую исследование по запросу: '{enhanced_query}'")
    
    return {
        **state,
        "query": enhanced_query,
        "steps": steps
    }

def node_search_papers(state: AgentState) -> AgentState:
    query = state["query"]
    papers_json = tools.search_papers(query, n_results=30)
    all_papers = json.loads(papers_json)
    
    if "error" in all_papers:
        steps = state.get("steps", [])
        steps.append(f"Ошибка поиска: {all_papers['error']}")
        return {
            **state,
            "search_results": [],
            "steps": steps
        }
    
    steps = state.get("steps", [])
    steps.append(f"Найдено {len(all_papers)} статей до фильтрации")
    
    return {
        **state,
        "search_results": all_papers,
        "steps": steps
    }

def node_filter_papers(state: AgentState) -> AgentState:
    all_papers = state["search_results"]
    
    if not all_papers:
        return {**state, "filtered_results": [], "filter_stats": {}}
    
    relevant_papers = []
    stats = {
        'total': len(all_papers),
        'filtered': 0,
        'by_category': {}
    }
    
    for paper in all_papers:
        if is_relevant_category(paper['categories']):
            paper['priority'] = get_category_priority(paper['categories'])
            relevant_papers.append(paper)
            
            cats = paper['categories'].split()
            for cat in cats:
                if cat in TARGET_CATEGORIES:
                    cat_name = TARGET_CATEGORIES[cat]
                    stats['by_category'][cat_name] = stats['by_category'].get(cat_name, 0) + 1
    
    relevant_papers.sort(key=lambda x: (x['priority'] * 0.3 + x['relevance'] * 0.7), reverse=True)

    filtered_papers = relevant_papers[:10]
    
    stats['filtered'] = len(filtered_papers)
    
    steps = state.get("steps", [])
    steps.append(f"Отфильтровано {len(filtered_papers)} релевантных статей из {len(all_papers)}")
    
    if stats['by_category']:
        top_cats = sorted(stats['by_category'].items(), key=lambda x: -x[1])[:3]
        steps.append(f"Основные категории: {', '.join([f'{cat} ({count})' for cat, count in top_cats])}")
    
    return {
        **state,
        "filtered_results": filtered_papers,
        "filter_stats": stats,
        "steps": steps
    }

def node_analyze_results(state: AgentState) -> AgentState:

    query = state["query"]
    papers = state["filtered_results"]
    
    if not papers:
        analysis = "По запросу не найдено релевантных статей"
    else:
        analysis = tools.analyze_results(query, papers)
    
    steps = state.get("steps", [])
    # steps.append(f"Проанализировал результаты")
    
    return {
        **state,
        "analysis": analysis,
        "steps": steps
    }

def node_generate_answer(state: AgentState) -> AgentState:
    original_query = state["original_query"]
    papers = state["filtered_results"]
    analysis = state["analysis"]
    stats = state.get("filter_stats", {})
    
    if not papers:
        answer = f"""По запросу "{original_query}" не найдено релевантных статей"""
    else:
        answer = f"""
Результаты:
   Запрос: {original_query}
{analysis}
Статьи:
"""
        
        for i, paper in enumerate(papers[:5], 1):
            
            answer += f"""
{i}. {paper['title']}
   - Релевантность: {paper['relevance']}
   - Категория: {paper['categories']}
   - Год: {paper['published'][:4]}
   - Кратко: {paper['abstract'][:200]}...
"""    
    steps = state.get("steps", [])
    #steps.append(f"Ответ сгенерирован")
    
    return {
        **state,
        "final_answer": answer,
        "steps": steps
    }

def create_research_agent():    
    # Инициализация графа
    workflow = StateGraph(AgentState)

    workflow.add_node("plan", node_plan_research)
    workflow.add_node("search", node_search_papers)
    workflow.add_node("filter", node_filter_papers)
    workflow.add_node("analyze", node_analyze_results)
    workflow.add_node("generate", node_generate_answer)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "filter")
    workflow.add_edge("filter", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_research_assistant(query: str):
    agent = create_research_agent()
    
    # Инициализация состояния
    initial_state = {
        "messages": [],
        "original_query": query,
        "query": query,
        "search_results": [],
        "filtered_results": [],
        "analysis": "",
        "final_answer": "",
        "steps": [],
        "filter_stats": {}
    }

    final_state = agent.invoke(initial_state)

    print("Лог выполнения:")
    for step in final_state["steps"]:
        print(f"  {step}")
    print(final_state["final_answer"])   
    return final_state

def interactive_agent():
    print("\nПримеры запросов:")
    print("  - neural networks transformers")
    print("  - knowledge distillation BERT")
    print("  - computer vision CNN")
    print("  - нейронные сети для текста")
    print("\nВведите 'quit' для выхода\n")
    
    while True:
        query = input("Ваш запрос: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        run_research_assistant(query)

if __name__ == "__main__":
    interactive_agent()