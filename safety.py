import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()

def check_safety(user_text):
    """
    Checks for Injection Attempts and Calls OpenAI's Moderation API to check if the input is safe.
    Returns: (is_safe: bool, category: str)
    """
    # --- Part 1: Logic/Jailbreak Filter ---
    injection_keywords = ["ignore all", "system prompt", "reveal", "secret key", "developer mode"]
    lower_text = user_text.lower()
    
    for word in injection_keywords:
        if word in lower_text:
            return False, "Potential System Manipulation attempt detected"

    # --- Part 2: OpenAI Moderation API ---
    client = OpenAI()
    
    try:
        response = client.moderations.create(input=user_text)
    except OpenAIError as exc:
        message = str(exc).lower()
        if any(keyword in message for keyword in [
            "invalid_api_key",
            "invalid key",
            "expired",
            "authentication",
            "missing",
            "api key",
            "openai key",
        ]):
            return False, "OpenAI API key issue: missing, invalid, or expired. Please verify OPENAI_API_KEY."
        raise
    output = response.results[0]

    # If 'flagged' is True, the content violated OpenAI's safety policies
    if output.flagged:
        # Find which category was triggered (e.g., 'harassment' or 'hate')
        violated_categories = [cat for cat, value in output.categories.__dict__.items() if value]
        return False, ", ".join(violated_categories)
    
    return True, None