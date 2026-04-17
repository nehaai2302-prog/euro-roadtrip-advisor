import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from tools import get_weather_forecast, estimate_fuel_cost, calculate_toll_vignette
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langsmith import traceable



# --- 1. SETUP VECTOR STORE ---

current_dir = Path(__file__).parent.absolute()
CHROMA_PATH = str(current_dir / "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


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
import json

def extract_tool_params(user_query: str, intent: str, chat_history: list):
    history_text = "\n".join(
    [f"{m['role']}: {m['content']}" for m in chat_history[-5:]]
  )
    prompt = f"""
You are a STRICT parameter extraction system.
Use conversation history to resolve references like:
- "there"
- "that city"
- "same place"
- "it"

Extract ONLY valid JSON for the given intent.
------------------------------------
IMPORTANT RULE:
If a city or destination is mentioned,
you MUST estimate realistic driving distance
from Munich, Germany.

Use approximate European driving distances.

------------------------------------

INTENT RULES:

weather:
{{"city": "string"}}

fuel:
{{"distance_km": number}}

toll:
{{"country": "string"}}

------------------------------------
EXAMPLES:

"weather in Berlin"
→ {{"city": "Berlin"}}

"fuel cost for trip to Berlin"
→ {{"distance_km": 580}}

"drive to Paris"
→ {{"distance_km": 850}}

"toll in Switzerland"
→ {{"country": "Switzerland"}}

------------------------------------
RULES:
- Return ONLY JSON
- No explanation
- No extra text
- If missing info, infer from context and estimate reasonably if possible
- If still unclear, return {{}} ONLY as last resort

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


# --- 4. MAIN FUNCTION ---
@traceable(name="Main RAG Pipeline")
def get_response(user_query: str, chat_history: list, traveler_type: str = "Standard", trip_context: dict = None):
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
        calculate_toll_vignette
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

    elif intent == "toll":
        params = extract_tool_params(user_query, "toll", chat_history)
        if not params or "country" not in params:
            return {
            "answer": "I couldn't determine the country for the toll calculation. Please specify the country.",
            "steps": []
            }
        result = calculate_toll_vignette.invoke(params)
        steps.append({"tool": "toll", "output": result})
        context = result

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
    - If the user asks for an itinerary, create a simple 1-day plan with 3-5 stops or stopovers based on the context and preferences.
    - In case of contradictions between user query and trip_context, always prioritize user query but mention the contradiction in the answer.
    - If the user asks for an itinerary but the intent was misclassified and you have RAG context, try to incorporate that context into a mini-itinerary or list of recommendations.
    - In case of preferenes that conflict (e.g., user wants to avoid highways but also prefers them), mention the conflict and provide a balanced recommendation.
    - Refuse anything unrelated.
    - If Intent was "chitchat", respond in a friendly and conversational manner without providing travel advice and don't use Context.
    """

    response = answer_llm.invoke(final_prompt)
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    
    return {
        "answer": response.content,
        "steps": steps,
        "usage": usage
    }