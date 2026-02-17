import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
import json

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

class AgentState(TypedDict):
    messages: List[Any]
    query: str
    search_results: List[Dict]
    analysis: str
    final_answer: str
    steps: List[str]

class ResearchTools:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chromadb_data")
        self.collection = self.client.get_collection("arxiv_papers")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def search_papers(self, query: str, n_results: int = 5) -> str:
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
                    'abstract': results['documents'][0][i][:300] + "..."
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
            cat = paper['categories'].split()[0] if paper['categories'] else "other"
            categories[cat] = categories.get(cat, 0) + 1
        
        analysis += "Распределение по категориям:\n"
        for cat, count in categories.items():
            analysis += f"  - {cat}: {count} статей\n"

        analysis += "\nНаиболее релевантные статьи:\n"
        for i, paper in enumerate(papers[:3], 1):
            analysis += f"\n{i}. {paper['title']}\n"
            analysis += f"   - Релевантность: {paper['relevance']}\n"
            analysis += f"   - Категории: {paper['categories']}\n"
            analysis += f"   - Аннотация: {paper['abstract'][:150]}...\n"
        
        return analysis

tools = ResearchTools()

def node_plan_research(state: AgentState) -> AgentState:
    query = state["query"]
    steps = state.get("steps", [])
    # steps.append(f" Планирую исследование по запросу: '{query}'")
    
    # добавить логику с LLM для улучшения запроса
    enhanced_query = query
    if len(query.split()) < 3:
        enhanced_query = query + " research paper"
        #steps.append(f" Улучшенный запрос: '{enhanced_query}'")
    
    return {
        **state,
        "query": enhanced_query,
        "steps": steps
    }

def node_search_papers(state: AgentState) -> AgentState:
    query = state["query"]
    papers_json = tools.search_papers(query, n_results=5)
    papers = json.loads(papers_json)
    
    if "error" in papers:
        steps = state.get("steps", [])
        steps.append(f" Ошибка поиска: {papers['error']}")
        return {
            **state,
            "search_results": [],
            "steps": steps
        }
    
    steps = state.get("steps", [])
    steps.append(f" Найдено {len(papers)} статей")
    
    return {
        **state,
        "search_results": papers,
        "steps": steps
    }

def node_analyze_results(state: AgentState) -> AgentState:
    query = state["query"]
    papers = state["search_results"]
    
    if not papers:
        analysis = "По запросу ничего не найдено. Попробуйте изменить формулировку"
    else:
        analysis = tools.analyze_results(query, papers)
    
    steps = state.get("steps", [])
    #steps.append(f" Проанализировал результаты")
    
    return {
        **state,
        "analysis": analysis,
        "steps": steps
    }

def node_generate_answer(state: AgentState) -> AgentState:
    query = state["query"]
    papers = state["search_results"]
    analysis = state["analysis"]
    
    if not papers:
        answer = f"""По запросу "{query}" не найдено статей"""
    else:
        answer = f"""
 Результаты:
   Запрос: {query}

{analysis}
 Найденные статьи:
"""
        
        for i, paper in enumerate(papers, 1):
            answer += f"""
{i}. {paper['title']}
   - Релевантность: {paper['relevance']}
   - Категории: {paper['categories']}
   - Год: {paper['published'][:4]}
   - Аннотация: {paper['abstract'][:200]}...
"""
    
    steps = state.get("steps", [])
    #steps.append(f" Сгенерировал ответ")
    
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
    workflow.add_node("analyze", node_analyze_results)
    workflow.add_node("generate", node_generate_answer)
    
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

def run_research_assistant(query: str):
    agent = create_research_agent()
    
    # Инициализация состояния
    initial_state = {
        "messages": [],
        "query": query,
        "search_results": [],
        "analysis": "",
        "final_answer": "",
        "steps": []
    }
    
    final_state = agent.invoke(initial_state)

    print(" Лог выполнения:")
    for step in final_state["steps"]:
        print(f"  {step}")
    print(final_state["final_answer"])
    
    return final_state

def interactive_agent():
    print("\n Примеры запросов:")
    print("  - neural networks deep learning")
    print("  - quantum computing")
    print("  - knowledge distillation")
    print("\nВведите 'quit' для выхода\n")
    
    while True:
        query = input(" Ваш запрос: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        run_research_assistant(query)

if __name__ == "__main__":
    interactive_agent()