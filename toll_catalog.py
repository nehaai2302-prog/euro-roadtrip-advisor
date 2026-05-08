"""
Approximate toll / vignette guidance per country (educational only).
Figures are illustrative order-of-magnitude; users must verify on official sites.
"""

COUNTRY_TOLL_RULES = {
    "DE": {
        "name": "Germany",
        "scheme": "no_general_car_vignette",
        "summary": (
            "Passenger cars generally pay no distance-based motorway vignette. "
            "Certain tunnels (e.g. Warnowtunnel) and truck toll (LKW-Maut) apply to trucks."
        ),
        "official_url": "https://www.bmvi.de/",
        "illustrative_note": None,
    },
    "AT": {
        "name": "Austria",
        "scheme": "vignette_digital_plus_sections",
        "summary": (
            "Motorways (Autobahnen) require a vignette (digital or sticker) by duration; "
            "several major corridors also charge separate section tolls on top of the vignette."
        ),
        "official_url": "https://www.asfinag.at/",
        "illustrative_note": (
            "Reference passenger-car tariffs below are illustrative; verify vehicle class, validity dates, "
            "and current prices on ASFINAG before travel."
        ),
        "vignettes": [
            {"type": "1-Day Digital", "price_eur": 9.60, "scope": "All Autobahns"},
            {"type": "10-Day Sticker/Digital", "price_eur": 12.80, "scope": "All Autobahns"},
            {"type": "2-Month Sticker/Digital", "price_eur": 32.00, "scope": "All Autobahns"},
            {"type": "Annual Sticker/Digital", "price_eur": 106.80, "scope": "All Autobahns"},
        ],
        "section_tolls": [
            {"road": "A13 Brenner", "type": "Section Toll", "price_eur": 12.50},
            {"road": "A10 Tauern", "type": "Section Toll", "price_eur": 15.00},
            {"road": "S16 Arlberg", "type": "Section Toll", "price_eur": 13.00},
            {"road": "A9 Gleinalm", "type": "Section Toll", "price_eur": 12.00},
        ],
    },
    "CH": {
        "name": "Switzerland",
        "scheme": "vignette_sticker",
        "summary": (
            "Motorway network requires an annual vignette for cars (single calendar-year sticker)."
        ),
        "official_url": "https://www.admin.ch/",
        "illustrative_note": "Annual vignette is typically around CHF 40 order of magnitude (verify yearly price).",
    },
    "FR": {
        "name": "France",
        "scheme": "distance_segment_autoroute",
        "summary": (
            "Péage applies when you use tolled autoroute segments (gates/tickets or tags), not simply because a stop lies in France; "
            "cost depends on exact entries/exits and vehicle class."
        ),
        "official_url": "https://www.autoroutes.fr/",
        "illustrative_note": (
            "Cross-border and near-border trips (e.g. Alsace / Strasbourg toward Germany) often stay on non-tolled national roads "
            "and may avoid péages unless the chosen itinerary uses autoroutes à péage. Routing apps disagree on corridors and "
            "this tool does not infer péage totals from city names—use an official autoroute calculator or operator site for your "
            "planned route. No single vignette covers all péages; urban zones and bridges can still add separate charges."
        ),
    },
    "IT": {
        "name": "Italy",
        "scheme": "distance_segment_autostrada",
        "summary": (
            "Main motorways are often tolled per segment (telepass or ticket). "
            "Urban ZTL/congestion charges are separate."
        ),
        "official_url": "https://www.autostrade.it/",
        "illustrative_note": "Total varies heavily by chosen exits and operator.",
    },
    "ES": {
        "name": "Spain",
        "scheme": "mixed",
        "summary": (
            "Many highways are free; some private concessions charge per segment."
        ),
        "official_url": "https://www.dgt.es/",
        "illustrative_note": None,
    },
    "PT": {
        "name": "Portugal",
        "scheme": "distance_segment",
        "summary": (
            "Several motorways use electronic tolls or Via Verde class-based charging."
        ),
        "official_url": "https://www.portugaltolls.pt/",
        "illustrative_note": None,
    },
    "PL": {
        "name": "Poland",
        "scheme": "mixed_concessions",
        "summary": (
            "Some motorway sections are tolled by private operators (e.g. parts of A1/A2/A4); "
            "many national roads remain toll-free."
        ),
        "official_url": "https://www.gov.pl/",
        "illustrative_note": "Corridor-specific; compare tolled vs toll-free alternatives.",
    },
    "CZ": {
        "name": "Czech Republic",
        "scheme": "vignette_digital",
        "summary": (
            "Motorway vignette required for cars on designated roads (electronic vignette)."
        ),
        "official_url": "https://www.czdoprava.cz/",
        "illustrative_note": "Short-duration vignettes are commonly modest single-digit to low tens EUR (verify current).",
    },
    "SK": {
        "name": "Slovakia",
        "scheme": "vignette_digital",
        "summary": (
            "Electronic vignette required for specified motorway sections."
        ),
        "official_url": "https://www.eznamka.sk/",
        "illustrative_note": None,
    },
    "SI": {
        "name": "Slovenia",
        "scheme": "vignette_digital",
        "summary": (
            "Vignette required for motorways and expressways for cars."
        ),
        "official_url": "https://evinjeta.dars.si/",
        "illustrative_note": None,
    },
    "HU": {
        "name": "Hungary",
        "scheme": "vignette_e_matrica",
        "summary": (
            "E-matrica vignette required for use of tolled road network."
        ),
        "official_url": "https://nemzetiutj.hu/",
        "illustrative_note": None,
    },
    "BE": {
        "name": "Belgium",
        "scheme": "mostly_free_some_concessions",
        "summary": (
            "Most motorways are free for cars; some tunnels/links may charge."
        ),
        "official_url": "https://mobilit.belgium.be/",
        "illustrative_note": None,
    },
    "NL": {
        "name": "Netherlands",
        "scheme": "mostly_free",
        "summary": (
            "Most roads have no vignette; specific tunnels (e.g. Westerscheldetunnel) may charge."
        ),
        "official_url": "https://www.government.nl/",
        "illustrative_note": None,
    },
    "LU": {
        "name": "Luxembourg",
        "scheme": "no_vignette",
        "summary": "No motorway vignette system for cars.",
        "official_url": "https://www.gouvernement.lu/",
        "illustrative_note": None,
    },
    "DK": {
        "name": "Denmark",
        "scheme": "bridges_tolls",
        "summary": (
            "No general vignette; major bridges (e.g. Storebælt, Øresund) have tolls."
        ),
        "official_url": "https://www.transportstyrelsen.dk/",
        "illustrative_note": None,
    },
    "SE": {
        "name": "Sweden",
        "scheme": "congestion_bridges",
        "summary": (
            "No vignette; bridge tolls and congestion charging exist in some cities."
        ),
        "official_url": "https://www.transportstyrelsen.se/",
        "illustrative_note": None,
    },
    "NO": {
        "name": "Norway",
        "scheme": "auto_pass",
        "summary": (
            "Many roads/tunnels/ferries use AutoPASS tolling; no classic vignette."
        ),
        "official_url": "https://www.autopass.no/",
        "illustrative_note": None,
    },
    "LT": {
        "name": "Lithuania",
        "scheme": "motorway_free_cars_commercial_e_vignette",
        "summary": (
            "Standard passenger cars and typical leisure camper vans within normal car/light-vehicle rules generally "
            "drive on Lithuanian motorways with no vignette or distance toll. Electronic road-user charges "
            "(e-momentinė-style vignettes) apply to commercial/heavy goods vehicles and certain larger buses—not "
            "ordinary road-trip cars."
        ),
        "official_url": "https://www.transport.lt/",
        "illustrative_note": (
            "Large motorhomes registered or classified as trucks, or above commercial weight thresholds, may fall under "
            "the heavy-vehicle scheme—confirm vehicle category with Lithuanian toll guidance before travel."
        ),
    },
    "HR": {
        "name": "Croatia",
        "scheme": "distance_segment_motorway",
        "summary": (
            "Main motorways (e.g. HAC network) are distance-priced: ticket on entry / ENC electronic toll for frequent users. "
            "Some bridges or tunnels may have separate tolls."
        ),
        "official_url": "https://hac.hr/",
        "illustrative_note": "Total depends on exact entries/exits and vehicle class; no single vignette covers all motorway kilometres.",
    },
}

DEFAULT_COUNTRY_RULE = {
    "name": None,
    "scheme": "unknown",
    "summary": (
        "Toll rules vary; check national motorway authority or tourism transport guidance "
        "for vignettes, electronic tolls, and motorway concessions."
    ),
    "official_url": None,
    "illustrative_note": None,
}


def rule_for_country(iso_code: str | None) -> dict:
    if not iso_code:
        out = DEFAULT_COUNTRY_RULE.copy()
        out["country_code"] = None
        out.setdefault("name", "Unknown")
        return out
    code = iso_code.strip().upper()
    if code in COUNTRY_TOLL_RULES:
        out = COUNTRY_TOLL_RULES[code].copy()
        out["country_code"] = code
        out.setdefault("name", code)
        return out
    out = DEFAULT_COUNTRY_RULE.copy()
    out["country_code"] = code
    out["name"] = code
    return out
