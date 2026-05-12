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
import re
import json
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from tools import get_weather_forecast, estimate_fuel_cost, calculate_route_and_tolls
from toll_estimate import extra_breakdown_lines
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

    q = (user_query or "").strip()
    q_l = q.lower()

    # Deterministic greeting/chitchat guard.
    # This prevents LLM routing drift for short social prompts like "hi" / "hello" / "how are you".
    chitchat_patterns = [
        r"^\s*(hi|hello|hey|hey there|hiya)\s*[!.?]*\s*$",
        r"^\s*(good\s+morning|good\s+afternoon|good\s+evening)\s*[!.?]*\s*$",
        r"^\s*(how\s+are\s+you|how\s+are\s+you\s+doing|how'?s\s+it\s+going)\s*[!.?]*\s*$",
        r"^\s*(thanks|thank\s+you|thx|bye|goodbye|see\s+you)\s*[!.?]*\s*$",
    ]
    if any(re.search(pat, q_l, re.I) for pat in chitchat_patterns):
        return {"intent": "chitchat"}

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
     "hi, hello, how are you, how's your day going, how's your day going?" → chitchat
     "thanks" → chitchat
     "good morning" → chitchat

    IMPORTANT:
    If the user asks for driving route/distance/duration between two places AND also asks for fuel or petrol cost in the SAME message,
    you MUST choose "route" (not "fuel"). The route tool can supply distance for the fuel estimate.

    GUARDRAILS:
    1. ONLY classify into one of the above intents.
    2. Use the chat history to understand context.


    Conversation:
    {history_text}

    Current query:
    {q}

    Respond ONLY in JSON format:
    {{"intent": "rag"}}
    """

    response = router_llm.invoke(prompt)

    try:
        return json.loads(response.content)
    except:
        return {"intent": "rag"}  # safe fallback
    
# ---3.1 HELPER: Extract parameters for tools ---

def _normalize_city_name(name: str) -> str:
    """Strip and title-case words so mUnich/munich match normal geocoding queries."""
    if not name:
        return ""
    s = " ".join(str(name).split())
    if not s:
        return ""
    return " ".join(part.capitalize() for part in s.split())


def _clean_city_fragment(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    s = s.strip("?.!,;:")
    s = re.sub(r"\s+", " ", s)
    return _normalize_city_name(s)


def heuristic_route_params(user_query: str) -> dict:
    """
    Fallback extraction when the LLM returns incomplete JSON.
    Handles 'from A to B', 'go to B from A', 'to B from A' (carefully), and 'between A and B'.
    """
    q = user_query or ""
    out = {}

    if re.search(r"\bfastest\b|\bquickest\b|\bfast\s+route\b", q, re.I):
        out["routing_mode"] = "fast"
    if re.search(r"\bshortest\b|\bshortest\s+(route|distance|path|way)\b", q, re.I):
        out["routing_mode"] = "short"

    # Require clause boundary OR end of query (queries without "?" must still match).
    boundary = r"(?=\s+and\s+(?:what|how|also)\b|\s+and\s+fuel\b|\?|$)"

    # "go to Vienna from Munich" → end, then start
    m = re.search(
        r"go\s+to\s+(.+?)\s+from\s+(.+?)" + boundary,
        q,
        re.I | re.DOTALL,
    )
    if m:
        out["end_city"] = _clean_city_fragment(m.group(1))
        out["start_city"] = _clean_city_fragment(m.group(2))
        return {k: v for k, v in out.items() if v}

    m = re.search(
        r"from\s+(.+?)\s+to\s+(.+?)" + boundary,
        q,
        re.I | re.DOTALL,
    )
    if m:
        out["start_city"] = _clean_city_fragment(m.group(1))
        out["end_city"] = _clean_city_fragment(m.group(2))
        return {k: v for k, v in out.items() if v}

    m = re.search(
        r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        q,
        re.I | re.DOTALL,
    )
    if m:
        out["start_city"] = _clean_city_fragment(m.group(1))
        out["end_city"] = _clean_city_fragment(m.group(2))
        return {k: v for k, v in out.items() if v}

    return {k: v for k, v in out.items() if v}


def merge_route_params(llm_params: dict, heuristic: dict, source_city: str) -> dict:
    p = {}
    for k, v in (llm_params or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        p[k] = v

    h = heuristic or {}
    for key in ("start_city", "end_city"):
        if key not in p or not str(p.get(key, "")).strip():
            if h.get(key):
                p[key] = h[key]

    if not p.get("routing_mode") and h.get("routing_mode"):
        p["routing_mode"] = h["routing_mode"]

    if (not p.get("start_city") or not str(p.get("start_city", "")).strip()) and source_city:
        p["start_city"] = _normalize_city_name(source_city)

    for key in ("start_city", "end_city"):
        if p.get(key):
            p[key] = _normalize_city_name(str(p[key]))

    rm = p.get("routing_mode", "fast")
    if rm not in ("short", "fast"):
        rm = "fast"
    p["routing_mode"] = rm
    return p


def query_mentions_tolls(user_query: str) -> bool:
    if not user_query:
        return False
    return bool(
        re.search(
            r"\btoll\b|vignette|maut|p[eé]age|pedaggio|autostrada|autostrady|autopista|"
            r"highway\s+(fee|charge)|road\s+charge",
            user_query,
            re.I,
        )
    )


def query_refers_previous_route(user_query: str) -> bool:
    if not user_query:
        return False
    q = user_query.strip()
    if re.search(
        r"\b(this|that|same|previous|last)\s+route\b|"
        r"\bthe\s+route\b|"
        r"\bthis\s+trip\b|\bthat\s+trip\b|\bsame\s+cities\b",
        q,
        re.I,
    ):
        return True
    contextual_followup = re.search(
        r"\bfuel\b|petrol|diesel|\bgas\b|\btoll\b|vignette|maut|route|trip|drive|distance|"
        r"\bcost\b|charg|charge|how\s+much|price|estimate|\bfee\b|fees",
        q,
        re.I,
    )
    vague_ref = re.search(
        r"\bthere\b|\bit\b|\bthem\b|\bsame\b\s+journey|\bsame\b\s+leg|\bthat\b\s+leg|those\s+cities",
        q,
        re.I,
    )
    return bool(contextual_followup and vague_ref)


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

Rules for route/toll:
- Extract BOTH cities even if the question also asks about fuel, tolls, or other topics. Ignore the extra questions for this JSON.
- Phrases like "to Vienna from Munich" mean end_city Vienna and start_city Munich.
- "from Munich to Vienna" means start_city Munich, end_city Vienna.
- City capitalization in the user message does not matter; still return proper city names.
- If the user says "fastest" or "quickest", routing_mode must be "fast". If "shortest distance", use "short".
- If routing preference is unclear, set routing_mode to "fast".

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
    if not checkpointer or not user_id:
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


def default_eval_trip_context() -> dict:
    """Neutral trip context for offline RAG evaluation (matches app shape)."""
    return {
        "traveler_type": "Standard",
        "trip_style": "Balanced",
        "source_city": "",
        "last_route": {},
        "preferences": {
            "avoid_tolls": False,
            "avoid_highways": False,
            "short_drives": False,
            "highways": False,
        },
    }


def _normalize_trip_context(trip_context: dict | None) -> dict:
    tc = dict(trip_context) if trip_context else {}
    prefs = tc.get("preferences") or {}
    return {
        "traveler_type": tc.get("traveler_type", "Standard"),
        "trip_style": tc.get("trip_style", "Balanced"),
        "source_city": str(tc.get("source_city", "") or "").strip(),
        "last_route": tc.get("last_route") or {},
        "preferences": {
            "avoid_tolls": bool(prefs.get("avoid_tolls", False)),
            "avoid_highways": bool(prefs.get("avoid_highways", False)),
            "short_drives": bool(prefs.get("short_drives", False)),
            "highways": bool(prefs.get("highways", False)),
        },
    }


def retrieve_rag_docs(user_query: str):
    """Retrieve knowledge-base chunks for RAG (same k as production retriever)."""
    return retriever.invoke(user_query)


def invoke_final_answer(
    user_query: str,
    context: str,
    chat_history: list,
    trip_context: dict | None,
    traveler_type: str,
    intent: str,
) -> tuple[str, dict]:
    """
    Build the production final prompt and return (answer_text, token_usage dict).
    Used for all intents; ``context`` is tool output, KB text, or empty (chitchat).
    """
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in (chat_history or [])[-5:]]
    )
    tc = _normalize_trip_context(trip_context)
    source_city = tc["source_city"]
    prefs = tc["preferences"]
    final_prompt = f"""
    You are a Euro Road Trip Advisor for {traveler_type} trips.

    Use the conversation history, provided context, and preferences to answer.

    Conversation History:
    {history_text}

    TRIP CONTEXT:
    - Traveler type: {tc.get('traveler_type')}
    - Trip style: {tc.get('trip_style')}
    - Source city (fallback): {source_city or "Not provided"}
    - Preferences:
    - Avoid tolls: {prefs['avoid_tolls']}
    - Avoid highways: {prefs['avoid_highways']}
    - Short drives: {prefs['short_drives']}
    - Prefer highways: {prefs['highways']}

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
    - For route duration in user-facing answers, present hours first (e.g., "about 3.5 hours"), with minutes in parentheses if useful.
    - If routing mode is short, explicitly explain that shortest minimizes distance, not travel time, and therefore can take longer than fastest routes.
    - If the user is traveling through the Alps (Austria, Switzerland, France, Italy), remind them that high-altitude tunnels and passes often charge extra fees in addition to the standard tolls or vignettes."
    - For route/toll questions, strictly use the tool output for distance, duration, and toll prices.
    - If the user asks for an itinerary, create a simple 1-day plan with 3-5 stops or stopovers based on the context and preferences.
    - In case of contradictions between user query and trip_context, always prioritize user query but mention the contradiction in the answer.
    - If the user asks for an itinerary but the intent was misclassified and you have RAG context, try to incorporate that context into a mini-itinerary or list of recommendations.
    - In case of preferenes that conflict (e.g., user wants to avoid highways but also prefers them), mention the conflict and provide a balanced recommendation.
    - Refuse anything unrelated.
    - For Poland tolls, recommend toll-free routes if the user wants to avoid tolls, but also mention that the A1,A2, A4 has a private toll section. If they want tolls, show the cost of A1 = 7€, A2 = 32€ or A4 = 8€ but don't calulate the total.
    - Do not attempt to calculate a "Total Estimated Toll" or sum up different highways. If a total is not explicitly provided in the retrieved data, state that the total cost depends on specific entry/exit points and recommend using an official calculator.
    - For real time information on Tolls, the user should be directed to official toll websites or apps. Always provide a disclaimer about checking current toll prices before traveling, especially for routes with known variability like those in Poland. 
    - If Intent was "chitchat", respond in a friendly and conversational manner without providing travel advice and don't use Context.
    """

    response = answer_llm.invoke(final_prompt)
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    return response.content, usage


def generate_rag_answer(
    user_query: str,
    retrieved_docs: list,
    chat_history: list,
    trip_context: dict | None,
    traveler_type: str = "Standard",
    *,
    intent: str = "rag",
) -> tuple[str, dict]:
    """
    Join retrieved LangChain documents into context and run the same final
    answer step as production RAG (for Ragas / offline eval).
    """
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return invoke_final_answer(
        user_query,
        context,
        chat_history,
        trip_context,
        traveler_type,
        intent,
    )


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

    source_city = (trip_context or {}).get("source_city", "").strip()

    # --- ROUTING ---
    route = route_query(user_query, chat_history)
    intent = route.get("intent", "rag")

    steps = []
    context = ""
    result = None
    last_route_update = None

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
        params = extract_tool_params(user_query, "fuel", chat_history) or {}
        merged_route = merge_route_params({}, heuristic_route_params(user_query), source_city)
        last_route = (trip_context or {}).get("last_route") or {}

        if query_refers_previous_route(user_query):
            if not str(merged_route.get("start_city", "")).strip() and str(last_route.get("start_city", "")).strip():
                merged_route["start_city"] = _normalize_city_name(str(last_route["start_city"]))
            if not str(merged_route.get("end_city", "")).strip() and str(last_route.get("end_city", "")).strip():
                merged_route["end_city"] = _normalize_city_name(str(last_route["end_city"]))

        if "distance_km" not in params or params.get("distance_km") is None:
            if merged_route.get("start_city") and merged_route.get("end_city"):
                routing_mode = merged_route.get("routing_mode", "fast")
                if trip_context and trip_context.get("trip_style") == "Fastest route":
                    routing_mode = "fast"
                result = calculate_route_and_tolls.invoke(
                    {
                        "start_city": merged_route["start_city"],
                        "end_city": merged_route["end_city"],
                        "avoid_tolls": bool((trip_context or {}).get("preferences", {}).get("avoid_tolls"))
                        if trip_context
                        else False,
                        "routing_mode": routing_mode,
                    }
                )
                steps.append({"tool": "route_and_tolls", "output": result})
                if isinstance(result, dict) and result.get("error"):
                    return {
                        "answer": f"I couldn't compute a route to estimate fuel: {result['error']}",
                        "steps": steps,
                    }
                params["distance_km"] = result["distance_km"]
            else:
                return {
                    "answer": "I couldn't determine the distance or cities for the fuel cost estimate. Please specify km or two cities (e.g. 'fuel cost Munich to Vienna').",
                    "steps": [],
                }

        fuel_result = estimate_fuel_cost.invoke({"distance_km": params["distance_km"]})
        steps.append({"tool": "fuel", "output": fuel_result})
        context = fuel_result

    elif intent in {"toll", "route"}:
        raw_params = extract_tool_params(user_query, "route", chat_history) or {}
        params = merge_route_params(raw_params, heuristic_route_params(user_query), source_city)
        last_route = (trip_context or {}).get("last_route") or {}

        if query_refers_previous_route(user_query):
            if not str(params.get("start_city", "")).strip() and str(last_route.get("start_city", "")).strip():
                params["start_city"] = _normalize_city_name(str(last_route["start_city"]))
            if not str(params.get("end_city", "")).strip() and str(last_route.get("end_city", "")).strip():
                params["end_city"] = _normalize_city_name(str(last_route["end_city"]))
            if not params.get("routing_mode") and last_route.get("routing_mode") in {"short", "fast"}:
                params["routing_mode"] = last_route["routing_mode"]

        if (
            "start_city" not in params
            or "end_city" not in params
            or not str(params.get("start_city", "")).strip()
            or not str(params.get("end_city", "")).strip()
        ):
            return {
                "answer": "I couldn't detect both start and end city. Try naming both cities explicitly (e.g. 'from Munich to Vienna'), spelling doesn't matter.",
                "steps": [],
            }

        routing_mode = params.get("routing_mode", "fast")
        if trip_context and trip_context.get("trip_style") == "Fastest route":
            routing_mode = "fast"
        result = calculate_route_and_tolls.invoke(
            {
                "start_city": params["start_city"],
                "end_city": params["end_city"],
                "avoid_tolls": bool((trip_context or {}).get("preferences", {}).get("avoid_tolls"))
                if trip_context
                else False,
                "routing_mode": routing_mode,
            }
        )
        steps.append({"tool": "route_and_tolls", "output": result})
        if isinstance(result, dict) and result.get("error"):
            context = result["error"]
        else:
            duration_min = result.get("duration_min", 0)
            try:
                duration_hours = round(float(duration_min) / 60, 1)
            except (TypeError, ValueError):
                duration_hours = 0.0
            toll_value = result.get("total_toll_eur")
            toll_text = f"EUR {toll_value}" if toll_value is not None else "Not available (segment/vignette systems vary)"
            bd = result.get("toll_breakdown_estimate") or []
            bd_lines = []
            for row in bd:
                nm = row.get("country_name") or row.get("country_code")
                line = f"  - {nm}: {row.get('summary', '')}"
                if row.get("illustrative_note"):
                    line += f" ({row['illustrative_note']})"
                bd_lines.append(line)
                bd_lines.extend(extra_breakdown_lines(row))
            bd_block = "\n".join(bd_lines) if bd_lines else "  (none inferred)"
            inf = result.get("toll_inference_note") or ""
            context = (
                f"Route: {result['start_city']} -> {result['end_city']}\n"
                f"Distance: {result['distance_km']} km\n"
                f"Duration: {duration_hours} hours ({duration_min} min)\n"
                f"Routing mode used: {routing_mode}\n"
                f"Total toll (numeric): {toll_text}\n"
                f"Countries inferred (sampled): {result.get('countries_inferred', [])}\n"
                f"Toll inference note: {inf}\n"
                f"Approximate toll guidance by country:\n{bd_block}\n"
                f"Toll confidence: {result.get('toll_confidence', 'unknown')}\n"
                f"Toll disclaimer: {result.get('toll_disclaimer', 'No disclaimer')}\n"
                f"Warnings: {result.get('warnings', [])}"
            )
            if routing_mode == "short":
                context += (
                    "\n\nThis route is shortest by distance, not necessarily fastest by time. "
                    "Shortest routes can include lower-speed roads, more local segments, urban traffic, "
                    "and intersections, so total travel time may be longer than a fastest-route option."
                )

            if re.search(r"\bfuel\b|petrol|diesel|gasoline|\bgas\b", user_query, re.I):
                fuel_line = estimate_fuel_cost.invoke({"distance_km": result["distance_km"]})
                steps.append({"tool": "fuel", "output": fuel_line})
                context += f"\n\nFuel estimate for this route distance:\n{fuel_line}"

    elif intent == "itinerary":
        tc = trip_context or {}
        style = tc.get("trip_style")
        prefs = tc.get("preferences") or {}
        itinerary_route = merge_route_params(
            extract_tool_params(user_query, "route", chat_history) or {},
            heuristic_route_params(user_query),
            source_city,
        )
        hist_blob = " ".join(
            m.get("content", "") for m in (chat_history or [])[-8:] if m.get("role") == "user"
        )
        itinerary_route = merge_route_params(
            itinerary_route,
            heuristic_route_params(hist_blob),
            source_city,
        )
        rm_it = itinerary_route.get("routing_mode") or "fast"
        if tc.get("trip_style") == "Fastest route":
            rm_it = "fast"
        sc2 = str(itinerary_route.get("start_city", "") or "").strip()
        ec2 = str(itinerary_route.get("end_city", "") or "").strip()
        if sc2 and ec2:
            last_route_update = {
                "start_city": _normalize_city_name(sc2),
                "end_city": _normalize_city_name(ec2),
                "routing_mode": rm_it if rm_it in ("short", "fast") else "fast",
            }
        steps.append({
            "tool": "itinerary_generator",
            "output": f"Generating {style} itinerary"
        })
        context = f"""
        Generate a {style} itinerary.
        Starting city (if user did not specify another city): {source_city or "Not provided"}
        Preferences:{prefs}
        """
    elif intent == "chitchat":
        context = ""
        steps.append({
        "tool": "none",
        "output": "General conversation"
        })

    else:  # RAG
        tc = trip_context or {}
        rag_route = merge_route_params(
            extract_tool_params(user_query, "route", chat_history) or {},
            heuristic_route_params(user_query),
            source_city,
        )
        hist_blob = " ".join(
            m.get("content", "") for m in (chat_history or [])[-8:] if m.get("role") == "user"
        )
        rag_route = merge_route_params(
            rag_route,
            heuristic_route_params(hist_blob),
            source_city,
        )
        rm_rag = rag_route.get("routing_mode") or "fast"
        if tc.get("trip_style") == "Fastest route":
            rm_rag = "fast"
        sc_r = str(rag_route.get("start_city", "") or "").strip()
        ec_r = str(rag_route.get("end_city", "") or "").strip()
        if sc_r and ec_r:
            last_route_update = {
                "start_city": _normalize_city_name(sc_r),
                "end_city": _normalize_city_name(ec_r),
                "routing_mode": rm_rag if rm_rag in ("short", "fast") else "fast",
            }

        docs = retrieve_rag_docs(user_query)

        chunks = "\n\n".join(doc.page_content for doc in docs)

        steps.append({
            "tool": "search_road_trip_knowledgebase",
            "output": chunks
        })

        context = chunks

    # --- FINAL ANSWER GENERATION ---
    answer_text, usage = invoke_final_answer(
        user_query,
        context,
        chat_history,
        trip_context,
        traveler_type,
        intent,
    )

    save_langgraph_checkpoint(
        user_id,
        {"last_query": user_query, "last_intent": intent, "last_answer": answer_text},
    )

    payload = {
        "answer": answer_text,
        "steps": steps,
        "usage": usage,
    }
    if last_route_update:
        payload["last_route_update"] = last_route_update
    if intent in {"toll", "route"} and isinstance(result, dict) and result.get("polyline"):
        show_toll_banner = intent == "toll" or query_mentions_tolls(user_query)
        payload["map_data"] = {
            "polyline": result["polyline"],
            "start_city": result["start_city"],
            "end_city": result["end_city"],
            "distance_km": result["distance_km"],
            "duration_min": result["duration_min"],
            "total_toll_eur": result["total_toll_eur"],
            "toll_confidence": result.get("toll_confidence"),
            "toll_disclaimer": result.get("toll_disclaimer"),
            "show_toll_banner": show_toll_banner,
            "toll_breakdown_estimate": result.get("toll_breakdown_estimate"),
            "countries_inferred": result.get("countries_inferred"),
            "routing_mode": routing_mode,
        }
    return payload