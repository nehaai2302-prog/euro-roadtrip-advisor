import folium
import flexpolyline as fp


def _emoji_pin_icon(emoji: str, accent: str) -> folium.DivIcon:
    """Pin marker using emoji only (no Leaflet PNG assets — works in Streamlit)."""
    html = (
        f'<div style="font-size:26px;line-height:34px;text-align:center;'
        f"width:40px;height:40px;border-radius:50%;border:3px solid {accent};"
        f'padding-top:2px;box-sizing:border-box;background:rgba(255,255,255,0.92);">{emoji}</div>'
    )
    return folium.DivIcon(html=html, icon_size=(40, 40), icon_anchor=(20, 38), class_name="emoji-pin-marker")


def render_route_map(polyline_str: str, start_city: str, end_city: str):
    if isinstance(polyline_str, list):
        coords = polyline_str
    else:
        coords = fp.decode(polyline_str)
    if not coords:
        return None

    m = folium.Map(location=coords[0], zoom_start=6, control_scale=True)
    folium.PolyLine(coords, color="blue", weight=4, opacity=0.9).add_to(m)
    folium.Marker(
        coords[0],
        tooltip=f"Start: {start_city}",
        icon=_emoji_pin_icon("📍", "#27ae60"),
    ).add_to(m)
    folium.Marker(
        coords[-1],
        tooltip=f"End: {end_city}",
        icon=_emoji_pin_icon("📍", "#e74c3c"),
    ).add_to(m)
    m.fit_bounds([coords[0], coords[-1]])
    return m
