"""
Infer countries along a route (polyline sampling + Nominatim reverse) and attach
approximate toll guidance from toll_catalog (educational only).
"""

from __future__ import annotations

import time
from typing import Any

import requests

from db import get_reverse_geo_country, set_reverse_geo_country
from toll_catalog import rule_for_country

USER_AGENT = "euro-road-trip-advisor/1.0"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
SLEEP_UNCACHED_REVERSE_S = 1.05
MAX_SAMPLES = 18


def _round_key(lat: float, lon: float) -> tuple[float, float]:
    return round(lat, 4), round(lon, 4)


def reverse_geocode_country(lat: float, lon: float) -> str | None:
    """Returns ISO 3166-1 alpha-2 country code or None."""
    lat_k, lon_k = _round_key(lat, lon)
    cached = get_reverse_geo_country(lat_k, lon_k)
    if cached is not None:
        return cached.upper() if cached else None

    time.sleep(SLEEP_UNCACHED_REVERSE_S)
    try:
        response = requests.get(
            NOMINATIM_REVERSE,
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 3,
                "addressdetails": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        addr = data.get("address") or {}
        cc = addr.get("country_code")
        if isinstance(cc, str) and cc.strip():
            code = cc.strip().upper()
            set_reverse_geo_country(lat_k, lon_k, code)
            return code
        set_reverse_geo_country(lat_k, lon_k, "")
        return None
    except Exception:
        set_reverse_geo_country(lat_k, lon_k, "")
        return None


def _sample_polyline_indices(n_coords: int, target: int = MAX_SAMPLES) -> list[int]:
    if n_coords <= 0:
        return []
    if n_coords <= target:
        return list(range(n_coords))
    step = max(1, (n_coords - 1) // (target - 1))
    indices = list(range(0, n_coords, step))
    if indices[-1] != n_coords - 1:
        indices.append(n_coords - 1)
    return sorted(set(indices))[:target]


def ordered_unique_countries(codes: list[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if not c:
            continue
        cu = c.upper()
        if cu not in seen:
            seen.add(cu)
            out.append(cu)
    return out


def countries_along_polyline(
    polyline_lat_lon: list[list[float]],
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> tuple[list[str], str]:
    """
    Returns (ordered_country_codes, inference_note).
    inference_note explains confidence caveats.
    """
    coords = polyline_lat_lon or []
    points: list[tuple[float, float]] = []

    points.append((start_lat, start_lon))
    for idx in _sample_polyline_indices(len(coords)):
        pt = coords[idx]
        if len(pt) >= 2:
            points.append((float(pt[0]), float(pt[1])))
    points.append((end_lat, end_lon))

    raw_codes: list[str | None] = []
    failures = 0
    for lat, lon in points:
        try:
            cc = reverse_geocode_country(lat, lon)
            raw_codes.append(cc)
            if cc is None:
                failures += 1
        except Exception:
            failures += 1
            raw_codes.append(None)

    ordered = ordered_unique_countries(raw_codes)
    if not ordered:
        note = "Country inference failed (no reverse-geocode results)."
    elif failures > len(points) // 2:
        note = "Partial country inference; some samples failed — verify corridor manually."
    elif len(ordered) == 1 and len(points) > 3:
        note = "Only one country detected along sampled points; cross-border corridor may be incomplete."
    else:
        note = "Countries inferred from sampled route points along the polyline."

    return ordered, note


def extra_breakdown_lines(row: dict[str, Any]) -> list[str]:
    """Human-readable lines for catalog vignettes and section tolls (reference tariffs)."""
    lines: list[str] = []
    for v in row.get("vignettes") or []:
        typ = v.get("type", "?")
        price = v.get("price_eur")
        scope = v.get("scope", "")
        if isinstance(price, (int, float)):
            lines.append(f"    • Vignette — {typ}: €{price:.2f} ({scope})")
        else:
            lines.append(f"    • Vignette — {typ}: {price} ({scope})")
    for s in row.get("section_tolls") or []:
        road = s.get("road", "?")
        typ = s.get("type", "")
        price = s.get("price_eur")
        if isinstance(price, (int, float)):
            lines.append(f"    • Section toll — {road} ({typ}): €{price:.2f}")
        else:
            lines.append(f"    • Section toll — {road} ({typ}): {price}")
    return lines


def build_toll_breakdown(country_codes: list[str]) -> list[dict[str, Any]]:
    breakdown: list[dict[str, Any]] = []
    for code in country_codes:
        rule = rule_for_country(code)
        breakdown.append(
            {
                "country_code": rule.get("country_code") or code,
                "country_name": rule.get("name") or code,
                "scheme": rule.get("scheme"),
                "summary": rule.get("summary"),
                "official_url": rule.get("official_url"),
                "illustrative_note": rule.get("illustrative_note"),
                "vignettes": rule.get("vignettes"),
                "section_tolls": rule.get("section_tolls"),
            }
        )
    return breakdown


def refresh_toll_breakdown_from_catalog(result: dict[str, Any]) -> dict[str, Any]:
    """
    Rebuild toll_breakdown_estimate from countries_inferred using the current toll_catalog.
    Use on cached routes so catalog edits propagate without re-running polyline reverse-geocode.
    """
    if result.get("error"):
        return result
    cc = result.get("countries_inferred")
    if isinstance(cc, list) and cc:
        result["toll_breakdown_estimate"] = build_toll_breakdown(cc)
    return result


def attach_toll_guidance(result: dict[str, Any]) -> dict[str, Any]:
    """
    Mutates and returns result with toll_breakdown_estimate, toll_disclaimer,
    toll_confidence, countries_inferred, toll_inference_note.
    Safe to call on cached payloads missing these fields.
    """
    if result.get("error"):
        return result

    if result.get("toll_guidance_attached"):
        return result

    poly = result.get("polyline")
    start = result.get("start") or {}
    end = result.get("end") or {}

    if not isinstance(poly, list) or not poly:
        result.setdefault(
            "toll_disclaimer",
            "Approximate toll guidance unavailable without route geometry.",
        )
        result.setdefault("toll_confidence", "low")
        result.setdefault("toll_breakdown_estimate", [])
        result["toll_guidance_attached"] = True
        return result

    try:
        o_lat = float(start["lat"])
        o_lon = float(start["lon"])
        d_lat = float(end["lat"])
        d_lon = float(end["lon"])
    except (KeyError, TypeError, ValueError):
        result.setdefault("toll_disclaimer", "Approximate toll guidance unavailable (missing endpoints).")
        result.setdefault("toll_confidence", "low")
        result.setdefault("toll_breakdown_estimate", [])
        result["toll_guidance_attached"] = True
        return result

    countries, inf_note = countries_along_polyline(poly, o_lat, o_lon, d_lat, d_lon)
    breakdown = build_toll_breakdown(countries)

    result["countries_inferred"] = countries
    result["toll_inference_note"] = inf_note
    result["toll_breakdown_estimate"] = breakdown

    if len(countries) >= 2:
        conf = "medium"
    elif countries:
        conf = "low"
    else:
        conf = "low"

    result["toll_confidence"] = conf
    result["toll_disclaimer"] = (
        "Approximate guidance only: vignettes, péages, and motorway tolls vary by exact corridor, "
        "vehicle class, and date. Always confirm on official operator sites before travel."
    )

    # Legacy field: keep empty list for segment tolls; numeric total not computed.
    result.setdefault("country_tolls", [])
    result.setdefault("total_toll_eur", None)

    result["toll_guidance_attached"] = True
    return result
