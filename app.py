import os
from dotenv import load_dotenv
import streamlit as st
import json
import pandas as pd
from datetime import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from streamlit_folium import st_folium

# Import your custom modules
from rag_pipeline_fast import estimate_cost
from safety import check_safety
from auth import get_authenticator
from db import init_db, load_history, save_message, load_preferences, save_preferences
from map_utils import render_route_map

# NOTE: Run python ingest.py first to build the vector database (./vectordb).
# After ingestion, you can launch the app with: streamlit run app.py
# The app uses the persisted DB and does not rebuild it.


# Load from .env locally
load_dotenv()

# Get API key from .env (local) or st.secrets (Streamlit Cloud)
api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in .env (local) or Streamlit Secrets (cloud).")
    st.stop()

# Set it for the entire process
os.environ["OPENAI_API_KEY"] = api_key


# 1. INITIAL SETUP
st.set_page_config(page_title="Euro Road Trip Advisor", page_icon="🚗", layout="wide")
init_db()

authenticator = get_authenticator()
name, authentication_status, username = authenticator.login("main")

if authentication_status is False:
    st.error("Invalid username or password.")
    st.stop()
if authentication_status is None:
    st.warning("Please log in to continue.")
    st.info("Default demo user: user1 / ChangeMe123!")
    st.stop()

if "user_id" not in st.session_state:
    st.session_state.user_id = username


# 2. CHAT MEMORY INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = load_history(username)
if "result" not in st.session_state:
    st.session_state.result = {}

# 2.1 JSON Export Function
def export_json(messages):
    return json.dumps({
        "created_at": str(datetime.now()),
        "conversation": messages
    }, indent=2)

# 2.2 CSV Export Function
def export_csv(messages):
    df = pd.DataFrame(messages)
    return df.to_csv(index=False)

# 2.3 PDF Export Function
def export_pdf(messages):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []

    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]

        elements.append(Paragraph(f"<b>{role}:</b> {content}", styles["Normal"]))
        elements.append(Spacer(1, 10))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# 3. SIDEBAR UI
with st.sidebar:
    st.title("🚗 Trip Control Panel")
    st.caption(f"Logged in as: **{name}**")
    authenticator.logout("Logout", "sidebar")

    # -------------------------
    # 1. TRAVEL PROFILE (HIGH VALUE)
    # -------------------------
    st.subheader("👥 Travelers")

    saved_trip_context = load_preferences(username) or {}
    traveler_options = ["Solo Traveler", "Couple", "Family with Kids", "Elderly Parents", "Friends"]
    trip_style_options = ["Balanced", "Fastest route", "Scenic route", "Budget-focused", "Relaxed"]

    traveler_type = st.selectbox(
        "Who is traveling?",
        traveler_options,
        index=traveler_options.index(saved_trip_context.get("traveler_type"))
        if saved_trip_context.get("traveler_type") in traveler_options else 0,
    )

    trip_style = st.radio(
        "Trip style",
        trip_style_options,
        index=trip_style_options.index(saved_trip_context.get("trip_style"))
        if saved_trip_context.get("trip_style") in trip_style_options else 0,
    )

    # -------------------------
    # 2. PREFERENCES (REAL VALUE)
    # -------------------------
    st.subheader("⚙️ Preferences")

    saved_prefs = saved_trip_context.get("preferences", {})
    avoid_tolls = st.checkbox("Avoid toll roads", value=bool(saved_prefs.get("avoid_tolls", False)))
    avoid_highways = st.checkbox("Prefer scenic roads", value=bool(saved_prefs.get("avoid_highways", False)))
    short_drives = st.checkbox("Limit long driving hours", value=bool(saved_prefs.get("short_drives", False)))
    highways = st.checkbox("Prefer highways for faster travel", value=bool(saved_prefs.get("highways", False)))
    
    # -------------------------
    trip_context = {
    "traveler_type": traveler_type,
    "trip_style": trip_style,
    "preferences": {
        "avoid_tolls": avoid_tolls,
        "avoid_highways": avoid_highways,
        "short_drives": short_drives,
        "highways": highways
        }
    }
    save_preferences(username, trip_context)
    
    # 3. SYSTEM INFO (OPTIONAL BUT USEFUL)
    # -------------------------
    st.subheader("📊 Session Info")

    st.write(f"Messages: {len(st.session_state.messages)}")

    # -------------------------

    # 4. USAGE & COST ESTIMATION (REAL VALUE)
    st.sidebar.subheader("💰 Usage")

    usage = st.session_state.result.get("usage", {})
    st.sidebar.write(f"Input tokens: {usage.get('prompt_tokens', 0)}")
    st.sidebar.write(f"Output tokens: {usage.get('completion_tokens', 0)}")

    cost = estimate_cost(usage)
    st.sidebar.write(f"Estimated cost: ${cost}")
    # -------------------------

    # 4. RESET
    # -------------------------
    st.divider()
    st.subheader("⬇️ Export Conversation")

    messages = st.session_state.messages

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button("📄", export_json(messages), "chat.json")
        st.caption("JSON")

    with col2:
        st.download_button("📊", export_csv(messages), "chat.csv")
        st.caption("CSV")

    with col3:
        st.download_button("📘", export_pdf(messages), "chat.pdf")
        st.caption("PDF")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🚗 Euro Road Trip Advisor")
st.caption(f"Currently advising for a **{traveler_type}** with **{trip_style}** trip.")
st.markdown("---")

# 4. DISPLAY CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 5. CHAT INPUT & LOGIC

if prompt := st.chat_input("Ask about your route, driving laws, or stopovers..."):
    
    # --- STEP A: SAFETY FILTER (The Gatekeeper) ---
    is_safe, reason = check_safety(prompt)
    
    if not is_safe:
        st.error(f"🚨 **Safety Alert:** Your message was flagged for: **{reason}**. Please keep your questions related to legal and safe travel.")
    
    else:  
        # --- STEP B: ADD USER MESSAGE TO UI ---
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(username, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- STEP C: EXECUTION (Agent-based Tools + RAG) ---
        with st.chat_message("assistant"):
            # We use st.status to show the Agent's reasoning process
            with st.status("🤖 AI Travel Advisor processing...", expanded=True) as status:
                # ✅ NEW: Limit memory size (prevents token overflow & improves speed)
                MAX_HISTORY = 10
                chat_history = st.session_state.messages[-MAX_HISTORY:]
                
                from rag_pipeline_fast import get_response
                
                # Execute the Agent-based RAG Pipeline. 
                # In 1.2.x, result is now a dictionary containing 'messages' and 'steps' (the modern equivalent of intermediate_steps).
                # The agent now decides if it needs weather, fuel, or the knowledge base
                result = get_response(
                    prompt,
                    chat_history=chat_history,
                    traveler_type=traveler_type,
                    trip_context=trip_context,
                    user_id=username,
                )
                st.session_state.result = result
                
                status.update(label="✅ Decision Made & Advice Generated", state="complete")

            # 1. Extract the Final Answer
            # The last message in the list is the AI's final response
            final_answer = result["answer"]
            st.markdown(final_answer)
            if result.get("map_data"):
                map_data = result["map_data"]
                route_map = render_route_map(
                    map_data["polyline"],
                    map_data["start_city"],
                    map_data["end_city"],
                )
                if route_map is not None:
                    st_folium(route_map, width=700, height=420, returned_objects=[])

            # 2. Display Evidence & Explainability (Intermediate Steps)
            with st.expander("🔍 System Logic: Tools & Knowledge Base Chunks"):
                if not result["steps"]:
                    st.write("The AI answered directly without needing extra tools.")
                else:
                    for step in result["steps"]:
                        # Step contains the 'action' (tool name) and 'observation' (result)
                        tool_name = step.get("tool", "Unknown Tool")
                        tool_output = step.get("output", "No data returned")
                        st.write(f"🛠️ **Tool Called:** `{tool_name}`")
                        
                        # If the RAG tool was used, show the retrieved text chunks
                        if tool_name == "search_road_trip_knowledgebase":
                            st.info("💡 **Chunks retrieved from Knowledge Base:**")
                            # This displays the text chunks returned by your retriever
                            st.markdown(tool_output)
                        else:
                            # This displays results from weather or fuel tools
                            st.success(f"📊 **Tool Output:** {tool_output}")

        # --- STEP D: STORE ASSISTANT MESSAGE ---
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        save_message(username, "assistant", final_answer)