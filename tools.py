import os
import requests
import random
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather_forecast(city: str) -> str:
    """
    Call this tool when the user asks about the current weather, temperature, 
    or climate conditions in a specific city or destination.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        # Fallback to simulation if no API key is found
        conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
        temp = random.randint(10, 25)
        return f"SIMULATED: {temp}°C and {random.choice(conditions)} in {city}"

    try:
        # Real API Call to OpenWeatherMap
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data["cod"] == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"The current weather in {city} is {temp}°C with {desc}."
        else:
            return f"Could not find weather for {city}. (Error: {data.get('message')})"
    except Exception as e:
        return f"Error connecting to weather service: {str(e)}"

@tool
def calculate_toll_vignette(country: str) -> str:
    """
    Use this tool to provide information on road taxes, vignettes, or tolls 
    required for driving in a specific European country.
    """
    tolls = {
        "Switzerland": "40 CHF (Annual Vignette Required)",
        "Austria": "11.50 EUR (10-Day Vignette)",
        "France": "Approx. €0.10/km (Péage System)",
        "Germany": "Free (No Tolls for cars)",
        "Norway": "AutoPASS (Electronic Tolls)",
        "Denmark": "Free (Tolls only on major bridges)"
    }
    return tolls.get(country, "Check local border for toll stickers and vignette requirements.")

@tool
def estimate_fuel_cost(distance_km: float) -> str:
    """
    Use this tool when the user asks for a math calculation of fuel expenses 
    or how much money they will spend on petrol/diesel for a specific distance.
    """
    # Logic for calculation
    avg_consumption_per_km = 0.07  # 7 liters per 100km
    avg_price_per_liter = 1.80     # Average Euro price
    
    total_liters = distance_km * avg_consumption_per_km
    total_cost = total_liters * avg_price_per_liter
    
    return f"Estimated fuel cost is €{total_cost:.2f} (consuming approx. {total_liters:.1f} liters)."