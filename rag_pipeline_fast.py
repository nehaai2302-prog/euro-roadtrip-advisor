import sys
try:
    # This will fail on your Windows machine (ImportError), which is fine!
    # It will succeed on Streamlit Linux.
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # On Windows, we just use the built-in sqlite3
    pass

import os
import json
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from tools import get_weather_forecast, estimate_fuel_cost, calculate_route_and_tolls
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.rate_limiters import InMemoryRateLimiter
from langsmith import traceable
import importlib


def _load_sqlite_saver():
    """
    Load SqliteSaver lazily to avoid static import resolution errors
    in environments where LangGraph sqlite extras are not installed.
    """
    try:
        module = importlib.import_module("langgraph.checkpoint.sqlite")
        return getattr(module, "SqliteSaver", None)
    except Exception:
        return None


SqliteSaver = _load_sqlite_saver()

# --- 1. SETUP VECTOR STORE ---

current_dir = Path(__file__).parent.absolute()
CHROMA_PATH = str(current_dir / "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

checkpointer = None
if SqliteSaver is not None:
    try:
        checkpointer = SqliteSaver.from_conn_string(str(current_dir / "checkpoints.sqlite"))
    except Exception:
        checkpointer = None


# --- 2. MODELS ---
# This limits the script to 2 request per second to stay within API quotas. (Requirement: Rate Limiting)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2.0, 
    max_bucket_size=10
)
# Fast model for routing
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, rate_limiter=rate_limiter).with_retry(stop_after_attempt=3)

# Model for final answer
answer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, rate_limiter=rate_limiter).with_retry(stop_after_attempt=3)

# --- 2.1 COST ESTIMATION ---
def estimate_cost(usage):
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    # Adjust if you change model pricing
    input_cost = input_tokens * 0.00000015
    output_cost = output_tokens * 0.00000060

    return round(input_cost + output_cost, 6)


# --- 3. ROUTER FUNCTION (MEMORY-AWARE) ---
def route_query(user_query: str, chat_history: list):
    """
    Classifies the user query into one intent using recent chat history.
    """

    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in chat_history[-5:]]
    )

    prompt = f"""
    You are a routing assistant.You are a strict intent classifier.

    IMPORTANT RULES:
    If the user asks to PLAN a trip (itinerary, 2+ days, visit, activities),
    you MUST choose "itinerary", NOT rag.
    
    If the message is a greeting, gratitude, or casual conversation,
    you MUST choose "chitchat", NOT rag.

    INTENTS:
    - weather → ONLY weather, temperature, forecast, climate, rain, sun
    - fuel → calculation of fuel cost, petrol, gas, cost of driving
    - toll → tolls, vignette, road charges
    - route → shortest route, fastest route, distance between two cities, driving duration
    - rag → driving laws, speed limits, safety rules, regulations Only.
    - itinerary → ANY request involving planning trips, schedules, activities, travel plans, sightseeing, 
      "3 days in X", "plan a trip", "what should I do in X","itinerary for X", "best stops in X", "where to visit in X", etc.
    - chitchat → greetings, thanks, small talk, general conversation
     "hello" → chitchat
     "thanks" → chitchat
     "good morning" → chitchat

    GUARDRAILS:
    1. ONLY classify into one of the above intents.
    2. Use the chat history to understand context.


    Conversation:
    {history_text}

    Current query:
    {user_query}

    Respond ONLY in JSON format:
    {{"intent": "rag"}}
    """

    response = router_llm.invoke(prompt)

    try:
        return json.loads(response.content)
    except:
        return {"intent": "rag"}  # safe fallback
    
# ---3.1 HELPER: Extract parameters for tools ---

def extract_tool_params(user_query: str, intent: str, chat_history: list):
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in chat_history[-5:]]
    )
    prompt = f"""
You are a strict JSON parameter extractor.
Use conversation history for references like "there", "it", "same city".
Return ONLY valid JSON and nothing else.

INTENT RULES:

weather:
{{"city": "string"}}

fuel:
{{"distance_km": number}}

toll:
route or toll:
{{"start_city": "string", "end_city": "string", "routing_mode": "short|fast"}}

Conversation history: {history_text}
Intent: {intent}
Query: {user_query}
"""

    response = router_llm.invoke(prompt)

    content = response.content.strip()

    try:
        data = json.loads(content)

        # 🔥 SAFETY LAYER (VERY IMPORTANT)
        if not isinstance(data, dict):
            return None

        return data

    except:
        return None


def save_langgraph_checkpoint(user_id: str, payload: dict):
    # Best-effort checkpoint write. Kept optional to avoid blocking responses.
    if not checkpointer:
        return
    try:
        checkpoint = {
            "v": 1,
            "id": str(uuid4()),
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": payload,
            "channel_versions": {},
            "versions_seen": {},
        }
        checkpointer.put(
            {"configurable": {"thread_id": user_id}},
            checkpoint,
            {},
            {},
        )
    except Exception:
        pass


# --- 4. MAIN FUNCTION ---
@traceable(name="Main RAG Pipeline")
def get_response(
    user_query: str,
    chat_history: list,
    traveler_type: str = "Standard",
    trip_context: dict = None,
    user_id: str = "anonymous",
):
    """
    Main pipeline:
    - Uses memory
    - Routes query
    - Calls tool or RAG
    - Generates final answer
    """

    from tools import (
        get_weather_forecast,
        estimate_fuel_cost,
        calculate_route_and_tolls,
    )

    # --- MEMORY (LIMITED) ---
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in chat_history[-5:]]
    )

    # --- ROUTING ---
    route = route_query(user_query, chat_history)
    intent = route.get("intent", "rag")

    steps = []
    context = ""

    # --- TOOL / RAG EXECUTION ---
    if intent == "weather":
        params = extract_tool_params(user_query, "weather", chat_history)
        if not params or "city" not in params:
            return {
            "answer": "I couldn't determine the city for the weather request. Please specify a city.",
            "steps": []
            }
        result = get_weather_forecast.invoke(params)
        steps.append({"tool": "weather", "output": result})
        context = result

    elif intent == "fuel":
        params = extract_tool_params(user_query, "fuel", chat_history)
        if not params or "distance_km" not in params:
            return {
            "answer": "I couldn't determine the distance for the fuel cost estimate. Please specify the distance.",
            "steps": []
            }
        result = estimate_fuel_cost.invoke(params)
        steps.append({"tool": "fuel", "output": result})
        context = result

    elif intent in {"toll", "route"}:
        params = extract_tool_params(user_query, "route", chat_history)
        if not params or "start_city" not in params or "end_city" not in params:
            return {
            "answer": "I couldn't detect both start and end city. Please provide them, e.g. 'Munich to Paris'.",
            "steps": []
            }
        routing_mode = params.get("routing_mode", "short")
        if trip_context and trip_context.get("trip_style") == "Fastest route":
            routing_mode = "fast"
        result = calculate_route_and_tolls.invoke(
            {
                "start_city": params["start_city"],
                "end_city": params["end_city"],
                "avoid_tolls": bool(trip_context["preferences"]["avoid_tolls"]) if trip_context else False,
                "routing_mode": routing_mode,
            }
        )
        steps.append({"tool": "route_and_tolls", "output": result})
        if isinstance(result, dict) and result.get("error"):
            context = result["error"]
        else:
            context = (
                f"Route: {result['start_city']} -> {result['end_city']}\n"
                f"Distance: {result['distance_km']} km\n"
                f"Duration: {result['duration_min']} min\n"
                f"Total toll: EUR {result['total_toll_eur']}\n"
                f"Country tolls: {result.get('country_tolls', [])}\n"
                f"Warnings: {result.get('warnings', [])}"
            )

    elif intent == "itinerary":
        style = trip_context.get("trip_style")
        steps.append({
        "tool": "itinerary_generator",
        "output": f"Generating {style} itinerary"
        })
        context = f"""
        Generate a {style} itinerary.
        Preferences:{trip_context['preferences']}
        """
    elif intent == "chitchat":
        context = ""
        steps.append({
        "tool": "none",
        "output": "General conversation"
        })

    else:  # RAG
        docs = retriever.invoke(user_query)

        chunks = "\n\n".join(doc.page_content for doc in docs)

        steps.append({
            "tool": "search_road_trip_knowledgebase",
            "output": chunks
        })

        context = chunks

    # --- FINAL ANSWER GENERATION ---
    final_prompt = f"""
    You are a Euro Road Trip Advisor for {traveler_type} trips.

    Use the conversation history, provided context, and preferences to answer.

    Conversation History:
    {history_text}

    TRIP CONTEXT:
    - Traveler type: {trip_context.get('traveler_type')}
    - Trip style: {trip_context.get('trip_style')}
    - Preferences:
    - Avoid tolls: {trip_context['preferences']['avoid_tolls']}
    - Avoid highways: {trip_context['preferences']['avoid_highways']}
    - Short drives: {trip_context['preferences']['short_drives']}
    - Prefer highways: {trip_context['preferences']['highways']}

    Context:
    {context}

    Question:
    {user_query}

    RULES:
    - If context is available → use it
    - Stay consistent with previous conversation
    - If the answer is not in the context, say you don't know, but try to be helpful with general knowledge.
    - Be concise and helpful. Don't be verbose.
    - Don't guess. 
    - Always respect preferences.
    - Adjust itinerary style based on trip_style.
    - If the user is traveling through the Alps (Austria, Switzerland, France, Italy), remind them that high-altitude tunnels and passes often charge extra fees in addition to the standard tolls or vignettes."
    - For route/toll questions, strictly use the tool output for distance, duration, and toll prices.
    - If the user asks for an itinerary, create a simple 1-day plan with 3-5 stops or stopovers based on the context and preferences.
    - In case of contradictions between user query and trip_context, always prioritize user query but mention the contradiction in the answer.
    - If the user asks for an itinerary but the intent was misclassified and you have RAG context, try to incorporate that context into a mini-itinerary or list of recommendations.
    - In case of preferenes that conflict (e.g., user wants to avoid highways but also prefers them), mention the conflict and provide a balanced recommendation.
    - Refuse anything unrelated.
    - For Poland tolls, recommend toll-free routes if the user wants to avoid tolls, but also mention that the A1,A2, A4 has a private toll section. If they want tolls, show the cost of A1 = 7€, A2 = 32€ or A4 = 8€ and that it's the fastest route.
    - For real time information on Tolls, the user should be directed to official toll websites or apps. Always provide a disclaimer about checking current toll prices before traveling, especially for routes with known variability like those in Poland. 
    - If Intent was "chitchat", respond in a friendly and conversational manner without providing travel advice and don't use Context.
    """

    response = answer_llm.invoke(final_prompt)
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    
    save_langgraph_checkpoint(
        user_id,
        {"last_query": user_query, "last_intent": intent, "last_answer": response.content},
    )

    payload = {
        "answer": response.content,
        "steps": steps,
        "usage": usage,
    }
    if intent in {"toll", "route"} and isinstance(result, dict) and result.get("polyline"):
        payload["map_data"] = {
            "polyline": result["polyline"],
            "start_city": result["start_city"],
            "end_city": result["end_city"],
            "distance_km": result["distance_km"],
            "duration_min": result["duration_min"],
            "total_toll_eur": result["total_toll_eur"],
        }
    return payload