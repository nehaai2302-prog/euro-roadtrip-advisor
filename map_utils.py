import folium
import flexpolyline as fp


def render_route_map(polyline_str: str, start_city: str, end_city: str):
    coords = fp.decode(polyline_str)
    if not coords:
        return None

    m = folium.Map(location=coords[0], zoom_start=6, control_scale=True)
    folium.PolyLine(coords, color="blue", weight=4, opacity=0.9).add_to(m)
    folium.Marker(coords[0], tooltip=f"Start: {start_city}").add_to(m)
    folium.Marker(coords[-1], tooltip=f"End: {end_city}").add_to(m)
    m.fit_bounds([coords[0], coords[-1]])
    return m
