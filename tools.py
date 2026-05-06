import os
import requests
import random
import hashlib
from langchain_core.tools import tool
from dotenv import load_dotenv
from db import get_geocode, set_geocode, get_route_cache, set_route_cache

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
    

def _cache_key(*parts: str) -> str:
    raw = "|".join(str(p).lower() for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def geocode_city(city: str):
    cached = get_geocode(city)
    if cached:
        return cached

    api_key = os.getenv("HERE_API_KEY")
    if not api_key:
        raise ValueError("HERE_API_KEY is not configured.")

    response = requests.get(
        "https://geocode.search.hereapi.com/v1/geocode",
        params={"q": city, "limit": 1, "apiKey": api_key},
        timeout=8,
    )
    response.raise_for_status()
    items = response.json().get("items", [])
    if not items:
        raise ValueError(f"Could not geocode city: {city}")
    position = items[0]["position"]
    lat, lon = position["lat"], position["lng"]
    set_geocode(city, lat, lon)
    return lat, lon


@tool
def calculate_route_and_tolls(
    start_city: str,
    end_city: str,
    avoid_tolls: bool = False,
    routing_mode: str = "short",
) -> dict:
    """Returns shortest/fastest route + toll breakdown from HERE Routing API."""
    if routing_mode not in {"short", "fast"}:
        routing_mode = "short"

    key = _cache_key(start_city, end_city, avoid_tolls, routing_mode)
    cached = get_route_cache(key)
    if cached:
        return cached

    o_lat, o_lon = geocode_city(start_city)
    d_lat, d_lon = geocode_city(end_city)

    params = {
        "transportMode": "car",
        "origin": f"{o_lat},{o_lon}",
        "destination": f"{d_lat},{d_lon}",
        "routingMode": routing_mode,
        "return": "summary,polyline,tolls",
        "tolls[summaries]": "total,country",
        "currency": "EUR",
        "vehicle[type]": "car",
        "lang": "en-US",
        "apiKey": os.getenv("HERE_API_KEY"),
    }
    if avoid_tolls:
        params["avoid[features]"] = "tollRoad"

    response = requests.get(
        "https://router.hereapi.com/v8/routes",
        params=params,
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()
    routes = data.get("routes", [])
    if not routes:
        return {"error": "No route found.", "distance_km": 0, "duration_min": 0, "total_toll_eur": 0}

    route = routes[0]
    section = route["sections"][0]
    summary = section.get("summary", {})
    toll_summary = route.get("tolls", {}).get("summary", {})

    result = {
        "start_city": start_city,
        "end_city": end_city,
        "distance_km": round(summary.get("length", 0) / 1000, 1),
        "duration_min": round(summary.get("duration", 0) / 60, 1),
        "total_toll_eur": round(toll_summary.get("total", {}).get("value", 0) or 0, 2),
        "country_tolls": [
            {
                "country": row.get("countryCode"),
                "eur": round(row.get("price", {}).get("value", 0) or 0, 2),
            }
            for row in toll_summary.get("country", [])
        ],
        "polyline": section.get("polyline"),
        "start": {"lat": o_lat, "lon": o_lon},
        "end": {"lat": d_lat, "lon": d_lon},
        "warnings": [n.get("title", n.get("code")) for n in route.get("notices", [])],
    }
    set_route_cache(key, result, ttl_days=7)
    return result


@tool
def calculate_toll_vignette(countries_data: list) -> str:
    """Backward-compatible wrapper that delegates to HERE route+toll calculation."""
    if not countries_data or len(countries_data) < 2:
        return "Please provide at least start and end cities to calculate tolls."
    start_city = countries_data[0].get("city") or countries_data[0].get("country")
    end_city = countries_data[-1].get("city") or countries_data[-1].get("country")
    if not start_city or not end_city:
        return "Please provide valid start and end cities."
    result = calculate_route_and_tolls.invoke(
        {"start_city": start_city, "end_city": end_city, "avoid_tolls": False, "routing_mode": "short"}
    )
    if isinstance(result, dict) and result.get("error"):
        return result["error"]
    return f"Estimated toll: €{result.get('total_toll_eur', 0)}"

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