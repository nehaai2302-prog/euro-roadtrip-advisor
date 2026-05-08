import os
import requests
import random
import hashlib
from langchain_core.tools import tool
from dotenv import load_dotenv
from db import get_geocode, set_geocode, get_route_cache, set_route_cache
from toll_estimate import attach_toll_guidance, extra_breakdown_lines, refresh_toll_breakdown_from_catalog

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


ORS_DIRECTIONS_GEOJSON_URL = (
    "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
)
ORS_REQUEST_TIMEOUT_S = 45


def _ors_error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
    except Exception:
        text = (response.text or "").strip()
        return text[:500] if text else (response.reason or "Unknown error")
    err = data.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        code = err.get("code")
        if msg and code is not None:
            return f"{msg} (ORS code {code})"
        return msg or str(err)
    if isinstance(err, str):
        return err
    return str(data)[:500]


def _ors_post_directions(api_key: str, body: dict) -> requests.Response:
    return requests.post(
        ORS_DIRECTIONS_GEOJSON_URL,
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=ORS_REQUEST_TIMEOUT_S,
    )


def geocode_city(city: str):
    cached = get_geocode(city)
    if cached:
        return cached

    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": city, "format": "json", "limit": 1},
        headers={"User-Agent": "euro-road-trip-advisor/1.0"},
        timeout=8,
    )
    response.raise_for_status()
    items = response.json()
    if not items:
        raise ValueError(f"Could not geocode city: {city}")
    lat, lon = float(items[0]["lat"]), float(items[0]["lon"])
    set_geocode(city, lat, lon)
    return lat, lon


@tool
def calculate_route_and_tolls(
    start_city: str,
    end_city: str,
    avoid_tolls: bool = False,
    routing_mode: str = "short",
) -> dict:
    """Returns shortest/fastest route + toll metadata from ORS + Nominatim."""
    if routing_mode not in {"short", "fast"}:
        routing_mode = "short"

    key = _cache_key(start_city, end_city, avoid_tolls, routing_mode)
    cached = get_route_cache(key)
    if cached:
        if not cached.get("toll_guidance_attached"):
            attach_toll_guidance(cached)
        else:
            refresh_toll_breakdown_from_catalog(cached)
        set_route_cache(key, cached, ttl_days=7)
        return cached

    o_lat, o_lon = geocode_city(start_city)
    d_lat, d_lon = geocode_city(end_city)

    ors_api_key = os.getenv("ORS_API_KEY")
    if not ors_api_key:
        return {
            "error": "ORS_API_KEY is not configured. Please add it to your environment.",
            "distance_km": 0,
            "duration_min": 0,
            "total_toll_eur": None,
            "toll_confidence": "low",
            "toll_disclaimer": "Toll pricing is unavailable because routing provider is not configured.",
        }

    body = {
        "coordinates": [[o_lon, o_lat], [d_lon, d_lat]],
        "preference": "shortest" if routing_mode == "short" else "recommended",
    }
    if avoid_tolls:
        body["options"] = {"avoid_features": ["tollways"]}

    used_shortest_fallback = False
    response = _ors_post_directions(ors_api_key, body)

    if response.status_code == 401:
        return {"error": "OpenRouteService API key is invalid or unauthorized.", "distance_km": 0, "duration_min": 0, "total_toll_eur": None}
    if response.status_code == 429:
        return {"error": "OpenRouteService rate limit reached. Please try again later.", "distance_km": 0, "duration_min": 0, "total_toll_eur": None}

    if response.status_code != 200:
        detail = _ors_error_detail(response)
        # ORS uses HTTP 404 both for bad URLs and for "no route" / snapping failures (see ORS error docs).
        if (
            response.status_code == 404
            and routing_mode == "short"
            and body.get("preference") == "shortest"
        ):
            retry_body = dict(body)
            retry_body["preference"] = "recommended"
            response = _ors_post_directions(ors_api_key, retry_body)
            used_shortest_fallback = True
            if response.status_code != 200:
                detail2 = _ors_error_detail(response)
                return {
                    "error": (
                        "OpenRouteService could not build this route with shortest-distance preference "
                        f"or with the fallback recommended profile. {detail2}"
                    ),
                    "distance_km": 0,
                    "duration_min": 0,
                    "total_toll_eur": None,
                }
        elif response.status_code != 200:
            return {
                "error": f"OpenRouteService error ({response.status_code}): {detail}",
                "distance_km": 0,
                "duration_min": 0,
                "total_toll_eur": None,
            }

    data = response.json()
    features = data.get("features", [])
    if not features:
        return {"error": "No route found.", "distance_km": 0, "duration_min": 0, "total_toll_eur": None}

    route = features[0]
    props = route.get("properties", {})
    summary = props.get("summary", {})
    geometry = route.get("geometry", {})
    coordinates = geometry.get("coordinates", [])
    latlon_coords = [[coord[1], coord[0]] for coord in coordinates if len(coord) >= 2]

    warnings = list(props.get("warnings") or [])
    if used_shortest_fallback:
        warnings.append(
            "OpenRouteService did not return a shortest-distance route for this corridor; "
            "showing recommended routing instead (often longer in km but reachable)."
        )

    result = {
        "start_city": start_city,
        "end_city": end_city,
        "provider": "openrouteservice",
        "distance_km": round(summary.get("distance", 0) / 1000, 1),
        "duration_min": round(summary.get("duration", 0) / 60, 1),
        "total_toll_e ur": None,
        "country_tolls": [],
        "polyline": latlon_coords,
        "start": {"lat": o_lat, "lon": o_lon},
        "end": {"lat": d_lat, "lon": d_lon},
        "warnings": warnings,
    }
    attach_toll_guidance(result)
    set_route_cache(key, result, ttl_days=7)
    return result


def format_toll_estimate_text(result: dict) -> str:
    """Human-readable approximate toll / vignette guidance from route tool output."""
    parts = []
    cc = result.get("countries_inferred") or []
    if cc:
        parts.append("Countries (sampled along route): " + ", ".join(cc))
    note = result.get("toll_inference_note")
    if note:
        parts.append(note)
    for row in result.get("toll_breakdown_estimate") or []:
        nm = row.get("country_name") or row.get("country_code") or "?"
        parts.append(f"- {nm}: {row.get('summary', '')}")
        if row.get("illustrative_note"):
            parts.append(f"  Illustrative: {row['illustrative_note']}")
        for line in extra_breakdown_lines(row):
            parts.append(line)
        if row.get("official_url"):
            parts.append(f"  Official info: {row['official_url']}")
    disc = result.get("toll_disclaimer")
    if disc:
        parts.append("")
        parts.append(disc)
    return "\n".join(parts) if parts else "No toll guidance available."


@tool
def calculate_toll_vignette(countries_data: list) -> str:
    """Approximate vignette / toll-system guidance for a city-to-city drive (no invoice-accurate totals)."""
    if not countries_data or len(countries_data) < 2:
        return "Please provide at least start and end cities to estimate toll context."
    start_city = countries_data[0].get("city") or countries_data[0].get("country")
    end_city = countries_data[-1].get("city") or countries_data[-1].get("country")
    if not start_city or not end_city:
        return "Please provide valid start and end cities."
    result = calculate_route_and_tolls.invoke(
        {"start_city": start_city, "end_city": end_city, "avoid_tolls": False, "routing_mode": "short"}
    )
    if isinstance(result, dict) and result.get("error"):
        return result["error"]
    return format_toll_estimate_text(result)

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