import os
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st
import json
import pandas as pd
from datetime import datetime
from io import BytesIO
from openai import OpenAIError
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from streamlit_folium import st_folium

# Import your custom modules
from rag_pipeline import (
    estimate_cost,
    get_response,
)
from safety import check_safety
from auth import (
    authenticate_user,
    register_user,
    request_password_reset,
    reset_password_with_token,
)
from db import (
    init_db,
    load_history,
    save_message,
    load_preferences,
    save_preferences,
    load_last_route,
    save_last_route,
)
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

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "guest_session_id" not in st.session_state:
    st.session_state.guest_session_id = f"guest-{uuid4().hex[:8]}"
if "active_user_id" not in st.session_state:
    st.session_state.active_user_id = st.session_state.guest_session_id


# 2. CHAT MEMORY INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "result" not in st.session_state:
    st.session_state.result = {}
if "last_route" not in st.session_state:
    st.session_state.last_route = {}
if "last_route_scope" not in st.session_state:
    st.session_state.last_route_scope = None


def _routing_scope_user_id() -> str:
    """Username when logged in; stable guest id when not."""
    return st.session_state.auth_user or st.session_state.guest_session_id


def _sync_last_route_from_db():
    scope = _routing_scope_user_id()
    if st.session_state.last_route_scope != scope:
        row = load_last_route(scope)
        st.session_state.last_route = row if row else {}
        st.session_state.last_route_scope = scope


def _persist_last_route_session(start_city, end_city, routing_mode="fast"):
    if not start_city or not end_city:
        return
    rm = routing_mode if routing_mode in ("short", "fast") else "fast"
    st.session_state.last_route = {
        "start_city": str(start_city).strip(),
        "end_city": str(end_city).strip(),
        "routing_mode": rm,
    }
    save_last_route(_routing_scope_user_id(), start_city, end_city, rm)


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


def _logout_user():
    st.session_state.auth_user = None
    st.session_state.active_user_id = st.session_state.guest_session_id
    st.session_state.messages = []
    st.session_state.last_route_scope = None
    st.session_state.last_route = {}
    st.rerun()


def _format_tool_output_for_display(tool_name: str, tool_output):
    """Avoid dumping huge coordinate lists (ORS polyline) into the debug panel."""
    if not isinstance(tool_output, dict):
        return tool_output
    if tool_name != "route_and_tolls" or "polyline" not in tool_output:
        return tool_output
    summary = {k: v for k, v in tool_output.items() if k != "polyline"}
    poly = tool_output.get("polyline")
    if isinstance(poly, list):
        summary["polyline_point_count"] = len(poly)
        summary["polyline_note"] = "Omitted from debug view; shown on map above."
    elif poly:
        summary["polyline_note"] = "Encoded geometry (truncated in UI if very long)."
    return summary


def _openai_key_error_message(exc: Exception) -> str | None:
    message = str(exc) or ""
    lower = message.lower()
    if any(keyword in lower for keyword in [
        "invalid_api_key",
        "invalid key",
        "expired",
        "authentication",
        "permission denied",
        "api key",
        "openai key",
        "missing",
    ]):
        return (
            "❌ OpenAI API key issue: your key is missing, invalid, or expired. "
            "Please set a valid `OPENAI_API_KEY` in your `.env` file or Streamlit Secrets."
        )
    if "rate limit" in lower or "too many requests" in lower:
        return "⚠️ OpenAI rate limit reached. Please wait a moment and try again."
    if "service unavailable" in lower or "timeout" in lower or "connection" in lower:
        return "⚠️ OpenAI service is unavailable right now. Please try again later."
    return None


def render_account_forms(key_prefix: str = "account"):
    auth_tab, signup_tab, reset_tab = st.tabs(["Sign in", "Create account", "Reset password"])

    with auth_tab:
        if st.session_state.auth_user:
            st.success(f"Logged in as: {st.session_state.auth_user}")
            if st.button("Logout", key=f"{key_prefix}_logout_btn"):
                _logout_user()
        else:
            login_user = st.text_input("Username", key=f"{key_prefix}_login_username")
            login_pass = st.text_input("Password", type="password", key=f"{key_prefix}_login_password")
            if st.button("Login", key=f"{key_prefix}_login_btn"):
                if authenticate_user(login_user, login_pass):
                    st.session_state.auth_user = login_user
                    st.session_state.active_user_id = login_user
                    st.session_state.messages = load_history(login_user)
                    st.session_state.last_route_scope = None
                    st.success("Logged in successfully.")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            st.caption("You can continue as guest without login.")

    with signup_tab:
        new_user = st.text_input("New username", key=f"{key_prefix}_signup_username")
        new_email = st.text_input("Email", key=f"{key_prefix}_signup_email")
        new_pass = st.text_input("New password", type="password", key=f"{key_prefix}_signup_password")
        new_pass2 = st.text_input("Confirm password", type="password", key=f"{key_prefix}_signup_password2")
        if st.button("Create account", key=f"{key_prefix}_create_account_btn"):
            if new_pass != new_pass2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(new_user, new_email, new_pass)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    with reset_tab:
        st.caption("Step 1: Request a reset token with your username or email.")
        reset_identifier = st.text_input("Username or email", key=f"{key_prefix}_reset_identifier")
        if st.button("Generate reset token", key=f"{key_prefix}_generate_reset_token_btn"):
            ok, msg, reset_token = request_password_reset(reset_identifier)
            if ok:
                st.success(msg)
                if reset_token:
                    st.warning("Development mode: copy this token now. It is shown only once.")
                    st.code(reset_token)
            else:
                st.error(msg)

        st.divider()
        st.caption("Step 2: Use token to set a new password.")
        token_input = st.text_input("Reset token", key=f"{key_prefix}_reset_token_input")
        new_reset_password = st.text_input("New password", type="password", key=f"{key_prefix}_new_reset_password")
        confirm_reset_password = st.text_input(
            "Confirm new password",
            type="password",
            key=f"{key_prefix}_confirm_reset_password",
        )
        if st.button("Reset password", key=f"{key_prefix}_reset_password_btn"):
            ok, msg = reset_password_with_token(token_input, new_reset_password, confirm_reset_password)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

_sync_last_route_from_db()

# 3. SIDEBAR UI
open_account_panel = False
supports_dialog = hasattr(st, "dialog")
with st.sidebar:
    st.title("🚗 Trip Control Panel")
    st.subheader("🔐 Account")
    if st.session_state.auth_user:
        st.success(f"Logged in as: {st.session_state.auth_user}")
        account_col, logout_col = st.columns([1.4, 1])
        with account_col:
            if st.button("Manage account", key="sidebar_account_btn"):
                open_account_panel = True
        with logout_col:
            if st.button("Logout", key="sidebar_logout_btn"):
                _logout_user()
    else:
        st.info("Guest mode (not signed in)")
        if st.button("Sign in", key="sidebar_login_btn"):
            open_account_panel = True
    st.divider()

    # -------------------------
    # 1. TRIP START (optional default for inferred routes / itinerary)
    # -------------------------
    st.subheader("🏁 Starting Location")
    source_city = st.text_input(
        "Starting point",
        key="sidebar_starting_point",
        placeholder="e.g. Munich",
    )

    # -------------------------
    # 2. TRAVEL PROFILE (HIGH VALUE)
    # -------------------------
    st.subheader("👥 Travellers")

    current_user = st.session_state.auth_user
    saved_trip_context = load_preferences(current_user) if current_user else {}
    if not saved_trip_context:
        saved_trip_context = {}
    traveler_options = ["Solo Traveller", "Couple", "Family with Kids", "Elderly Parents", "Friends"]
    trip_style_options = ["Balanced", "Fastest route", "Scenic route", "Budget-focused", "Relaxed"]

    traveler_type = st.selectbox(
        "Who is travelling?",
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
    # 3. PREFERENCES (REAL VALUE)
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
    "source_city": source_city.strip(),
    "last_route": st.session_state.last_route,
    "preferences": {
        "avoid_tolls": avoid_tolls,
        "avoid_highways": avoid_highways,
        "short_drives": short_drives,
        "highways": highways
        }
    }
    if current_user:
        save_preferences(current_user, trip_context)
    
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

if supports_dialog:
    @st.dialog("Manage account")
    def show_account_dialog():
        render_account_forms("dialog")

    if open_account_panel:
        show_account_dialog()
else:
    with st.sidebar.expander("Account", expanded=False):
        render_account_forms("sidebar_expander")

if not st.session_state.auth_user:
    st.info("Start chatting instantly in guest mode, or log in to save your conversations and preferences across sessions.")
st.markdown("---")

# 4. DISPLAY CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 5. CHAT INPUT & LOGIC

if prompt := st.chat_input("Ask about your route, driving laws, or stopovers..."):
    
    # --- STEP A: SAFETY FILTER (The Gatekeeper) ---
    try:
        is_safe, reason = check_safety(prompt)
    except Exception as exc:
        user_error = _openai_key_error_message(exc)
        st.error(user_error or "❌ Unable to validate your message because of an OpenAI API issue. Please verify your OPENAI_API_KEY.")
        st.stop()
    
    if not is_safe:
        st.error(f"🚨 **Safety Alert:** Your message was flagged for: **{reason}**. Please keep your questions related to legal and safe travel.")
    
    else:  
        # --- STEP B: ADD USER MESSAGE TO UI ---
        st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.auth_user:
            save_message(st.session_state.auth_user, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        MAX_HISTORY = 10

        # --- STEP C: EXECUTION (Agent-based Tools + RAG) ---
        with st.chat_message("assistant"):
            # We use st.status to show the Agent's reasoning process
            with st.status("🤖 AI Travel Advisor processing...", expanded=True) as status:
                # ✅ NEW: Limit memory size (prevents token overflow & improves speed)
                chat_history = st.session_state.messages[-MAX_HISTORY:]

                # Execute the Agent-based RAG Pipeline. 
                # In 1.2.x, result is now a dictionary containing 'messages' and 'steps' (the modern equivalent of intermediate_steps).
                # The agent now decides if it needs weather, fuel, or the knowledge base
                try:
                    result = get_response(
                        prompt,
                        chat_history=chat_history,
                        traveler_type=traveler_type,
                        trip_context=trip_context,
                        user_id=st.session_state.auth_user,
                    )
                except Exception as exc:
                    user_error = _openai_key_error_message(exc)
                    st.error(user_error or "❌ Unable to generate a response because of an OpenAI API issue. Please verify your OPENAI_API_KEY.")
                    st.stop()

                st.session_state.result = result
                
                status.update(label="✅ Decision Made & Advice Generated", state="complete")

            # 1. Extract the Final Answer
            # The last message in the list is the AI's final response
            final_answer = result["answer"]
            st.markdown(final_answer)
            if result.get("map_data"):
                map_data = result["map_data"]
                _persist_last_route_session(
                    map_data.get("start_city"),
                    map_data.get("end_city"),
                    map_data.get("routing_mode", "fast"),
                )
                if map_data.get("show_toll_banner"):
                    if map_data.get("toll_disclaimer"):
                        confidence = map_data.get("toll_confidence", "unknown")
                        st.info(f"Toll confidence: {confidence}. {map_data['toll_disclaimer']}")
                    breakdown = map_data.get("toll_breakdown_estimate") or []
                    if breakdown:
                        with st.expander("Approximate toll / vignette guidance by country"):
                            for row in breakdown:
                                title = row.get("country_name") or row.get("country_code") or "Country"
                                st.markdown(f"**{title}** — {row.get('summary', '')}")
                                if row.get("illustrative_note"):
                                    st.caption(row["illustrative_note"])
                                for v in row.get("vignettes") or []:
                                    pe = v.get("price_eur")
                                    pe_s = f"€{float(pe):.2f}" if isinstance(pe, (int, float)) else str(pe)
                                    st.markdown(
                                        f"- **{v.get('type', '')}**: {pe_s} — {v.get('scope', '')}"
                                    )
                                for s in row.get("section_tolls") or []:
                                    pe = s.get("price_eur")
                                    pe_s = f"€{float(pe):.2f}" if isinstance(pe, (int, float)) else str(pe)
                                    st.markdown(
                                        f"- **{s.get('road', '')}** ({s.get('type', '')}): {pe_s}"
                                    )
                                if row.get("official_url"):
                                    st.markdown(f"[Official info]({row['official_url']})")
                route_map = render_route_map(
                    map_data["polyline"],
                    map_data["start_city"],
                    map_data["end_city"],
                )
                if route_map is not None:
                    st_folium(route_map, width=700, height=420, returned_objects=[])

            lr_upd = result.get("last_route_update")
            if isinstance(lr_upd, dict) and lr_upd.get("start_city") and lr_upd.get("end_city"):
                _persist_last_route_session(
                    lr_upd["start_city"],
                    lr_upd["end_city"],
                    lr_upd.get("routing_mode", "fast"),
                )

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
                            display_out = _format_tool_output_for_display(tool_name, tool_output)
                            if isinstance(display_out, dict):
                                st.success("📊 **Tool output** (summary; large fields omitted)")
                                st.json(display_out)
                            else:
                                st.success(f"📊 **Tool Output:** {display_out}")

        # --- STEP D: STORE ASSISTANT MESSAGE ---
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        if st.session_state.auth_user:
            save_message(st.session_state.auth_user, "assistant", final_answer)