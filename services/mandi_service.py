"""
Nearby tomato mandi discovery.

Resolves GPS or a place name, finds CEDA tomato markets in nearby
districts, computes distance and a price-based rating, then sorts.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from services import ceda_service


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mandi"
NEARBY_RADIUS_KM = 220.0
MIN_NEARBY_DISTRICTS = 8
MAX_NEARBY_DISTRICTS = 12
DEFAULT_SORT = "distance"

_district_coords: Optional[Dict[str, Dict[str, Any]]] = None
_market_coords: Optional[Dict[str, Dict[str, float]]] = None
_seed_markets: Optional[List[Dict[str, Any]]] = None


def _log(message: str) -> None:
    print(f"[MANDI SERVICE] {message}")


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    earth_radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_place(value: str) -> str:
    aliases = {
        "bengaluru": "bangalore",
        "bengalooru": "bangalore",
        "belagavi": "belgaum",
        "vijayapura": "bijapur",
        "kalaburagi": "gulbarga",
        "ballari": "bellary",
        "mysuru": "mysore",
        "shivamogga": "shimoga",
        "tumakuru": "tumkur",
        "mangaluru": "mangalore",
        "hubballi": "hubli",
    }
    text = " ".join(_clean_text(value).lower().replace("-", " ").split())
    return aliases.get(text, text)


def _estimate_travel_time(distance_km: float) -> str:
    if distance_km <= 0:
        return "0 min"
    minutes = int(round((distance_km / 45.0) * 60))
    minutes = max(minutes, 1)
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    remaining = minutes % 60
    if remaining == 0:
        return f"{hours} hr"
    return f"{hours} hr {remaining} min"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _district_coord_map() -> Dict[str, Dict[str, Any]]:
    global _district_coords
    if _district_coords is None:
        data = _load_json(DATA_DIR / "district_coords.json", {})
        _district_coords = data if isinstance(data, dict) else {}
    return _district_coords


def _market_coord_map() -> Dict[str, Dict[str, float]]:
    global _market_coords
    if _market_coords is None:
        data = _load_json(DATA_DIR / "market_coords.json", {})
        _market_coords = data if isinstance(data, dict) else {}
    return _market_coords


def _persist_market_coords() -> None:
    if _market_coords is None:
        return
    _save_json(DATA_DIR / "market_coords.json", _market_coords)


def _load_seed_markets() -> List[Dict[str, Any]]:
    global _seed_markets
    if _seed_markets is None:
        data = _load_json(DATA_DIR / "tomato_market_seed.json", [])
        _seed_markets = data if isinstance(data, list) else []
    return _seed_markets


def _geocode_place(query: str) -> Optional[Dict[str, Any]]:
    query = _clean_text(query)
    if not query:
        return None

    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": query,
                "count": 8,
                "language": "en",
                "format": "json",
            },
            timeout=12,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    except Exception as exc:
        _log(f"Forward geocode failed for '{query}': {exc}")
        return None

    india = [
        item
        for item in results
        if str(item.get("country_code") or "").upper() in {"IN", "IND"}
        or str(item.get("country") or "").lower() == "india"
    ]
    chosen = (india or results)[0] if (india or results) else None
    if not chosen:
        return None

    return {
        "name": _clean_text(chosen.get("name"), query),
        "admin1": _clean_text(chosen.get("admin1")),
        "admin2": _clean_text(chosen.get("admin2")),
        "latitude": _to_float(chosen.get("latitude"), float("nan")),
        "longitude": _to_float(chosen.get("longitude"), float("nan")),
    }


def _reverse_geocode(latitude: float, longitude: float) -> Dict[str, str]:
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "jsonv2",
                "zoom": 10,
                "addressdetails": 1,
            },
            headers={"User-Agent": "Agrinova-Backend/1.0 (nearby-mandis)"},
            timeout=12,
        )
        response.raise_for_status()
        address = response.json().get("address") or {}
    except Exception as exc:
        _log(f"Reverse geocode failed: {exc}")
        return {}

    return {
        "name": _clean_text(
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("county")
            or address.get("state_district")
        ),
        "district": _clean_text(
            address.get("state_district")
            or address.get("county")
            or address.get("city")
        ),
        "state": _clean_text(address.get("state")),
    }


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("Invalid coordinates.")
    if latitude < -90 or latitude > 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if longitude < -180 or longitude > 180:
        raise ValueError("Longitude must be between -180 and 180.")


def resolve_origin(
    latitude: Optional[float],
    longitude: Optional[float],
    location_name: Optional[str],
) -> Dict[str, Any]:
    place = _clean_text(location_name)
    has_coords = (
        latitude is not None
        and longitude is not None
        and math.isfinite(float(latitude))
        and math.isfinite(float(longitude))
    )

    if place and not has_coords:
        geo = _geocode_place(f"{place}, India") or _geocode_place(place)
        if geo is None:
            raise ValueError(
                f"Could not find a location named '{place}'. Try a city or district name."
            )
        _validate_coordinates(geo["latitude"], geo["longitude"])
        return {
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "label": ", ".join(
                part for part in [geo["name"], geo["admin1"]] if part
            ) or place,
            "state": geo["admin1"],
            "district": geo["admin2"] or geo["name"],
            "source": "location_name",
        }

    if not has_coords:
        raise ValueError("Provide GPS coordinates or a location name.")

    lat = float(latitude)
    lng = float(longitude)
    _validate_coordinates(lat, lng)
    details = _reverse_geocode(lat, lng)
    label = place or details.get("name") or f"{lat:.4f}, {lng:.4f}"
    if details.get("state") and details["state"] not in label:
        label = f"{label}, {details['state']}" if details.get("name") else details["state"]

    return {
        "latitude": lat,
        "longitude": lng,
        "label": label,
        "state": details.get("state") or "",
        "district": details.get("district") or details.get("name") or "",
        "source": "gps" if not place else "location_name",
    }


def _geography_key(state_id: int, district_id: int) -> str:
    return f"{state_id}:{district_id}"


def _match_geographies(name: str, field: str) -> List[Dict[str, Any]]:
    needle = _normalize_place(name)
    if not needle:
        return []
    matches: List[Dict[str, Any]] = []
    for geo in ceda_service.list_geographies():
        value = _normalize_place(str(geo.get(field) or ""))
        if value == needle or needle in value or value in needle:
            matches.append(geo)
    return matches


def _coords_for_district(geo: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    state_id = _to_int(geo.get("census_state_id"))
    district_id = _to_int(geo.get("census_district_id"))
    cached = _district_coord_map().get(_geography_key(state_id, district_id))
    if cached:
        lat = _to_float(cached.get("latitude"), float("nan"))
        lng = _to_float(cached.get("longitude"), float("nan"))
        if math.isfinite(lat) and math.isfinite(lng):
            return lat, lng

    district = _clean_text(geo.get("census_district_name"))
    state = _clean_text(geo.get("census_state_name"))
    result = _geocode_place(f"{district}, {state}, India")
    if result is None:
        return None

    coords = {
        "name": district,
        "state": state,
        "latitude": result["latitude"],
        "longitude": result["longitude"],
    }
    _district_coord_map()[_geography_key(state_id, district_id)] = coords
    _save_json(DATA_DIR / "district_coords.json", _district_coord_map())
    return result["latitude"], result["longitude"]


def _nearby_districts(origin: Dict[str, Any]) -> List[Dict[str, Any]]:
    geographies = ceda_service.list_geographies()
    origin_state = _normalize_place(origin.get("state") or "")
    origin_district = _normalize_place(origin.get("district") or origin.get("label") or "")
    preferred_state_ids = {
        _to_int(item.get("census_state_id"))
        for item in _match_geographies(origin_state, "census_state_name")
    }

    if preferred_state_ids:
        for geo in geographies:
            if _to_int(geo.get("census_state_id")) in preferred_state_ids:
                _coords_for_district(geo)

    ranked: List[Dict[str, Any]] = []
    for geo in geographies:
        state_id = _to_int(geo.get("census_state_id"))
        district_id = _to_int(geo.get("census_district_id"))
        cached = _district_coord_map().get(_geography_key(state_id, district_id))
        if not cached:
            continue
        lat = _to_float(cached.get("latitude"), float("nan"))
        lng = _to_float(cached.get("longitude"), float("nan"))
        if not math.isfinite(lat) or not math.isfinite(lng):
            continue
        distance = haversine_distance_km(
            origin["latitude"],
            origin["longitude"],
            lat,
            lng,
        )
        ranked.append({**geo, "latitude": lat, "longitude": lng, "distance": distance})

    if not ranked:
        return []

    ranked.sort(
        key=lambda item: (
            0
            if origin_district
            and _normalize_place(item.get("census_district_name") or "") == origin_district
            else 1,
            0 if _to_int(item.get("census_state_id")) in preferred_state_ids else 1,
            item["distance"],
        )
    )

    nearby = [item for item in ranked if item["distance"] <= NEARBY_RADIUS_KM]
    if len(nearby) < MIN_NEARBY_DISTRICTS:
        nearby = ranked[:MIN_NEARBY_DISTRICTS]
    return nearby[:MAX_NEARBY_DISTRICTS]


def _coords_for_market(
    name: str,
    city: str,
    state: str,
    fallback: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    cache_key = _normalize_place(f"{name}|{city}|{state}")
    cached = _market_coord_map().get(cache_key)
    if cached:
        return _to_float(cached["latitude"]), _to_float(cached["longitude"])

    queries = [
        f"{name} mandi, {city}, {state}, India",
        f"{name}, {city}, {state}, India",
        f"{name} APMC, {state}, India",
    ]
    for query in queries:
        result = _geocode_place(query)
        if result is None:
            continue
        lat, lng = result["latitude"], result["longitude"]
        _market_coord_map()[cache_key] = {"latitude": lat, "longitude": lng}
        _persist_market_coords()
        return lat, lng

    if fallback is not None:
        _market_coord_map()[cache_key] = {
            "latitude": fallback[0],
            "longitude": fallback[1],
        }
        _persist_market_coords()
        return fallback

    raise ValueError(f"Unable to geocode market {name}")


def _stable_rating(name: str, city: str) -> float:
    digest = hashlib.sha256(f"{name}|{city}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 21
    return round(3.6 + bucket / 20.0, 1)


def _price_ratings(markets: List[Dict[str, Any]]) -> None:
    priced = [item for item in markets if _to_float(item.get("price")) > 0]
    if not priced:
        for item in markets:
            item["rating"] = _stable_rating(item["name"], item["city"])
        return

    prices = [item["price"] for item in priced]
    low = min(prices)
    high = max(prices)
    spread = high - low if high > low else 1.0

    for item in markets:
        price = _to_float(item.get("price"))
        base = _stable_rating(item["name"], item["city"])
        if price <= 0:
            item["rating"] = base
            continue
        quality = (price - low) / spread
        trend_bonus = 0.15 if _to_float(item.get("trend")) > 0 else 0.0
        item["rating"] = round(min(5.0, max(3.4, 3.5 + quality * 1.3 + trend_bonus)), 1)


def _trend_from_prices(history: List[Dict[str, Any]], market_id: int) -> float:
    rows = [
        row
        for row in history
        if _to_int(row.get("market_id"), -1) == market_id and _to_float(row.get("modal_price")) > 0
    ]
    rows.sort(key=lambda row: str(row.get("date") or ""))
    if len(rows) < 2:
        return 0.0
    previous = _to_float(rows[-2].get("modal_price"))
    latest = _to_float(rows[-1].get("modal_price"))
    if previous <= 0:
        return 0.0
    return round(((latest - previous) / previous) * 100, 1)


def _collect_ceda_markets(districts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    commodity_id = ceda_service.tomato_commodity_id()
    collected: List[Dict[str, Any]] = []
    seen: set[tuple] = set()

    districts_by_state: Dict[int, List[Dict[str, Any]]] = {}
    for district in districts:
        state_id = _to_int(district.get("census_state_id"))
        districts_by_state.setdefault(state_id, []).append(district)

    for state_id, state_districts in districts_by_state.items():
        district_ids = [_to_int(item.get("census_district_id")) for item in state_districts]
        price_rows = ceda_service.list_prices(commodity_id, state_id, district_ids)
        latest = {}
        for row in price_rows:
            market_id = _to_int(row.get("market_id"), -1)
            if market_id < 0:
                continue
            current = latest.get(market_id)
            if current is None or str(row.get("date") or "") > str(current.get("date") or ""):
                latest[market_id] = row

        for district in state_districts:
            district_id = _to_int(district.get("census_district_id"))
            markets = ceda_service.list_markets(commodity_id, state_id, district_id)
            fallback = (district["latitude"], district["longitude"])
            city = _clean_text(district.get("census_district_name"), "Karnataka")
            state = _clean_text(district.get("census_state_name"), "Karnataka")

            for market in markets:
                name = _clean_text(
                    market.get("market_name")
                    or market.get("name")
                    or market.get("market")
                )
                if not name:
                    continue
                market_id = _to_int(market.get("market_id"), -1)
                key = (_normalize_place(name), _normalize_place(city))
                if key in seen:
                    continue
                seen.add(key)

                try:
                    lat, lng = _coords_for_market(name, city, state, fallback)
                except ValueError:
                    lat, lng = fallback

                price_row = latest.get(market_id, {})
                collected.append(
                    {
                        "name": name,
                        "city": city,
                        "state": state,
                        "district": city,
                        "market_id": market_id if market_id >= 0 else None,
                        "latitude": lat,
                        "longitude": lng,
                        "price": _to_float(price_row.get("modal_price")),
                        "trend": _trend_from_prices(price_rows, market_id),
                        "crops": ["Tomato"],
                        "source": "CEDA / AGMARKNET",
                    }
                )

    return collected


def _collect_seed_markets(
    origin: Dict[str, Any],
    existing: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    existing_keys = {
        (_normalize_place(item["name"]), _normalize_place(item["city"]))
        for item in existing
    }
    extras: List[Dict[str, Any]] = []
    for market in _load_seed_markets():
        name = _clean_text(market.get("name"))
        city = _clean_text(market.get("city"))
        if not name:
            continue
        key = (_normalize_place(name), _normalize_place(city))
        if key in existing_keys:
            continue
        lat = _to_float(market.get("latitude"), float("nan"))
        lng = _to_float(market.get("longitude"), float("nan"))
        if not math.isfinite(lat) or not math.isfinite(lng):
            continue
        distance = haversine_distance_km(
            origin["latitude"],
            origin["longitude"],
            lat,
            lng,
        )
        extras.append(
            {
                "name": name,
                "city": city,
                "state": _clean_text(market.get("state"), "Karnataka"),
                "district": city,
                "market_id": None,
                "latitude": lat,
                "longitude": lng,
                "price": 0.0,
                "trend": 0.0,
                "crops": ["Tomato"],
                "source": "CEDA / AGMARKNET",
                "_seed_distance": distance,
            }
        )

    extras.sort(key=lambda item: item["_seed_distance"])
    if not extras:
        return extras
    nearby = [item for item in extras if item["_seed_distance"] <= NEARBY_RADIUS_KM]
    chosen = nearby if nearby else extras[:15]
    for item in chosen:
        item.pop("_seed_distance", None)
    return chosen


def _finalize_markets(
    origin: Dict[str, Any],
    markets: List[Dict[str, Any]],
    sort_by: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen: set[tuple] = set()

    for market in markets:
        lat = _to_float(market.get("latitude"), float("nan"))
        lng = _to_float(market.get("longitude"), float("nan"))
        if not math.isfinite(lat) or not math.isfinite(lng) or (lat == 0 and lng == 0):
            continue
        key = (
            _normalize_place(market["name"]),
            _normalize_place(market["city"]),
            round(lat, 4),
            round(lng, 4),
        )
        if key in seen:
            continue
        seen.add(key)
        distance = haversine_distance_km(
            origin["latitude"],
            origin["longitude"],
            lat,
            lng,
        )
        market["distance"] = round(distance, 1)
        market["travel"] = _estimate_travel_time(distance)
        results.append(market)

    _price_ratings(results)

    if sort_by == "rating":
        results.sort(
            key=lambda item: (
                -_to_float(item.get("rating")),
                item.get("distance", float("inf")),
                item["name"].lower(),
            )
        )
    else:
        results.sort(
            key=lambda item: (
                item.get("distance", float("inf")),
                -_to_float(item.get("rating")),
                item["name"].lower(),
            )
        )

    return results


async def get_nearby_mandis(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location: Optional[str] = None,
    sort_by: str = DEFAULT_SORT,
) -> List[Dict[str, Any]]:
    origin = resolve_origin(latitude, longitude, location)
    districts = _nearby_districts(origin)
    _log(
        f"Origin {origin['label']} ({origin['latitude']:.4f}, {origin['longitude']:.4f}); "
        f"{len(districts)} nearby districts"
    )

    markets = _collect_ceda_markets(districts)
    ceda_live = bool(markets)
    if not markets:
        markets = _collect_seed_markets(origin, [])
    else:
        markets.extend(_collect_seed_markets(origin, markets))

    return _finalize_markets(origin, markets, sort_by)


async def get_nearby_mandi_response(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location: Optional[str] = None,
    sort_by: str = DEFAULT_SORT,
) -> Dict[str, Any]:
    requested_sort = "rating" if _clean_text(sort_by).lower() == "rating" else "distance"
    origin = resolve_origin(latitude, longitude, location)
    districts = _nearby_districts(origin)
    markets = _collect_ceda_markets(districts)
    ceda_live = bool(markets)
    if not markets:
        markets = _collect_seed_markets(origin, [])
    else:
        markets.extend(_collect_seed_markets(origin, markets))
    mandis = _finalize_markets(origin, markets, requested_sort)

    distances = [item["distance"] for item in mandis]
    states = sorted({item.get("state") or "" for item in mandis if item.get("state")})

    return {
        "mandis": mandis,
        "mandi_count": len(mandis),
        "radius_km": NEARBY_RADIUS_KM,
        "region": origin.get("state") or (states[0] if len(states) == 1 else "Nearby"),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "search_mode": origin["source"],
        "sorted_by": requested_sort,
        "sort_order": (
            "highest_to_lowest" if requested_sort == "rating" else "nearest_to_farthest"
        ),
        "farthest_distance_km": max(distances) if distances else 0,
        "searched_latitude": origin["latitude"],
        "searched_longitude": origin["longitude"],
        "location_label": origin["label"],
        "source": "CEDA / AGMARKNET",
        "ceda_live": ceda_live,
    }


def get_mandi_status() -> Dict[str, Any]:
    seed = _load_seed_markets()
    geographies = ceda_service.list_geographies()
    return {
        "database_exists": bool(geographies or seed),
        "database_path": str(DATA_DIR),
        "total_markets": len(seed),
        "markets_with_coordinates": sum(
            1
            for item in seed
            if _to_float(item.get("latitude")) and _to_float(item.get("longitude"))
        ),
        "markets_without_coordinates": 0,
        "ceda_api_configured": ceda_service.has_api_key(),
        "geography_count": len(geographies),
    }
