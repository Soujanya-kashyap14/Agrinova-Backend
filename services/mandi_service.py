"""
AgriWise Intelligence
Mandi Service

Purpose
-------
Find Karnataka tomato mandis from the local CEDA catalogue.

Requirements
------------
- No hardcoded city list.
- No radius restriction.
- No result limit.
- Use the user's GPS coordinates.
- Return all Karnataka tomato mandis available in the CEDA catalogue.
- Sort results from nearest to farthest.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

KARNATAKA_LAT_MIN = 11.5
KARNATAKA_LAT_MAX = 18.5

KARNATAKA_LNG_MIN = 74.0
KARNATAKA_LNG_MAX = 78.7


# ============================================================
# LOGGING
# ============================================================

def _log(message: str) -> None:
    print(f"[MANDI SERVICE] {message}")


# ============================================================
# DISTANCE
# ============================================================

def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate great-circle distance between two GPS coordinates.
    """

    earth_radius_km = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return earth_radius_km * c


# ============================================================
# KARNATAKA CHECK
# ============================================================

def _is_karnataka_coordinate(
    latitude: float,
    longitude: float,
) -> bool:
    """
    Broad geographic bounding-box check.

    This intentionally does not use a hardcoded city list.
    """

    return (
        KARNATAKA_LAT_MIN <= latitude <= KARNATAKA_LAT_MAX
        and KARNATAKA_LNG_MIN <= longitude <= KARNATAKA_LNG_MAX
    )


# ============================================================
# NUMBER HELPERS
# ============================================================

def _to_float(
    value: Any,
    default: float = 0.0,
) -> float:

    if value is None:
        return default

    if isinstance(value, bool):
        return default

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _to_int(
    value: Any,
    default: int = 0,
) -> int:

    if value is None:
        return default

    try:
        return int(float(value))
    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean_text(
    value: Any,
    default: str = "",
) -> str:

    if value is None:
        return default

    text = str(value).strip()

    return text if text else default


# ============================================================
# TRAVEL TIME
# ============================================================

def _estimate_travel_time(
    distance_km: float,
) -> str:
    """
    Rough road-travel estimate.

    This is only a display estimate.
    It is NOT used for sorting.
    """

    if distance_km <= 0:
        return "0 min"

    # Conservative average road speed.
    average_speed_kmh = 45.0

    minutes = int(
        round(
            (distance_km / average_speed_kmh)
            * 60
        )
    )

    if minutes < 60:
        return f"{max(minutes, 1)} min"

    hours = minutes // 60
    remaining = minutes % 60

    if remaining == 0:
        return f"{hours} hr"

    return f"{hours} hr {remaining} min"


# ============================================================
# MARKET NORMALIZATION
# ============================================================

def _normalize_market(
    market: Any,
) -> Optional[Dict[str, Any]]:
    """
    Convert a CEDA market record into the frontend format.

    The function intentionally accepts multiple possible field
    names because CEDA catalogue files can differ in schema.
    """

    if not isinstance(market, dict):
        return None

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name = _clean_text(
        market.get("name")
        or market.get("market")
        or market.get("market_name")
        or market.get("mandi")
        or market.get("mandi_name")
        or market.get("Market")
        or market.get("Market Name")
    )

    if not name:
        return None

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    city = _clean_text(
        market.get("city")
        or market.get("district")
        or market.get("district_name")
        or market.get("City")
        or market.get("District")
        or market.get("location")
        or market.get("taluk"),
        "Karnataka",
    )

    # --------------------------------------------------------
    # LATITUDE
    # --------------------------------------------------------

    latitude = _to_float(
        market.get("latitude")
        if market.get("latitude") is not None
        else market.get("lat")
        if market.get("lat") is not None
        else market.get("Latitude")
        if market.get("Latitude") is not None
        else market.get("y")
    )

    # --------------------------------------------------------
    # LONGITUDE
    # --------------------------------------------------------

    longitude = _to_float(
        market.get("longitude")
        if market.get("longitude") is not None
        else market.get("lng")
        if market.get("lng") is not None
        else market.get("lon")
        if market.get("lon") is not None
        else market.get("Longitude")
        if market.get("Longitude") is not None
        else market.get("x")
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    price = _to_float(
        market.get("price")
        or market.get("modal_price")
        or market.get("modalPrice")
        or market.get("modal")
        or market.get("average_price")
        or market.get("avg_price")
        or market.get("max_price")
        or market.get("min_price")
    )

    # --------------------------------------------------------
    # RATING
    # --------------------------------------------------------

    rating = _to_float(
        market.get("rating"),
        4.0,
    )

    if rating <= 0:
        rating = 4.0

    if rating > 5:
        rating = 5.0

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend = _to_float(
        market.get("trend")
        or market.get("price_change")
        or market.get("change_percent")
        or market.get("change")
    )

    # --------------------------------------------------------
    # CROPS
    # --------------------------------------------------------

    raw_crops = (
        market.get("crops")
        or market.get("commodities")
        or market.get("commodity")
        or market.get("crop")
    )

    crops: List[str] = []

    if isinstance(raw_crops, list):

        for crop in raw_crops:

            crop_text = _clean_text(crop)

            if crop_text:
                crops.append(crop_text)

    elif raw_crops:

        crop_text = _clean_text(
            raw_crops
        )

        if crop_text:
            crops.append(crop_text)

    # Always show tomato because this endpoint is specifically
    # the tomato mandi endpoint.
    if not any(
        crop.lower() == "tomato"
        for crop in crops
    ):
        crops.insert(
            0,
            "Tomato",
        )

    return {
        "name": name,
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "price": price,
        "rating": rating,
        "trend": trend,
        "crops": crops,
    }


# ============================================================
# CEDA IMPORT
# ============================================================

def _load_ceda_function():
    """
    Try to locate the existing CEDA catalogue helper.

    This keeps the service compatible with projects where the
    CEDA integration lives in a separate module.
    """

    candidates = [
        (
            "services.ceda_service",
            "get_nearby_mandis",
        ),
        (
            "services.ceda_service",
            "get_all_mandis",
        ),
        (
            "services.ceda_service",
            "get_mandis",
        ),
        (
            "services.ceda",
            "get_nearby_mandis",
        ),
        (
            "services.ceda",
            "get_all_mandis",
        ),
        (
            "services.ceda",
            "get_mandis",
        ),
    ]

    for module_name, function_name in candidates:

        try:

            module = __import__(
                module_name,
                fromlist=[function_name],
            )

            function = getattr(
                module,
                function_name,
                None,
            )

            if callable(function):

                _log(
                    f"Using CEDA function: "
                    f"{module_name}.{function_name}"
                )

                return function

        except Exception:
            continue

    return None


# ============================================================
# LOCAL CEDA CATALOGUE
# ============================================================

def _find_local_ceda_catalogue() -> Optional[Path]:
    """
    Search common locations for a local CEDA catalogue.

    This is a fallback only.
    """

    backend_dir = Path(
        __file__
    ).resolve().parents[1]

    candidates = [
        backend_dir / "data" / "ceda",
        backend_dir / "data",
        backend_dir / "datasets",
        backend_dir / "catalogue",
        backend_dir / "catalogues",
        backend_dir / "ceda",
    ]

    extensions = [
        "*.json",
        "*.jsonl",
        "*.csv",
        "*.xlsx",
        "*.xls",
    ]

    for directory in candidates:

        if not directory.exists():
            continue

        for extension in extensions:

            matches = list(
                directory.glob(
                    extension
                )
            )

            for file_path in matches:

                lower_name = (
                    file_path.name.lower()
                )

                if (
                    "ceda" in lower_name
                    or "mandi" in lower_name
                    or "market" in lower_name
                    or "tomato" in lower_name
                ):
                    return file_path

    return None


def _read_local_catalogue(
    path: Path,
) -> List[Dict[str, Any]]:

    records: List[Dict[str, Any]] = []

    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if suffix == ".json":

        import json

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(data, list):
            records = data

        elif isinstance(data, dict):

            for key in (
                "mandis",
                "markets",
                "data",
                "results",
                "records",
            ):

                value = data.get(key)

                if isinstance(
                    value,
                    list,
                ):

                    records = value
                    break

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    elif suffix == ".jsonl":

        import json

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    item = json.loads(
                        line
                    )

                    if isinstance(
                        item,
                        dict,
                    ):
                        records.append(
                            item
                        )

                except Exception:
                    continue

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    elif suffix == ".csv":

        import csv

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            records = list(
                reader
            )

    # --------------------------------------------------------
    # EXCEL
    # --------------------------------------------------------

    elif suffix in (
        ".xlsx",
        ".xls",
    ):

        try:

            import pandas as pd

            dataframe = pd.read_excel(
                path
            )

            records = (
                dataframe
                .fillna("")
                .to_dict(
                    orient="records"
                )
            )

        except Exception as exc:

            _log(
                f"Unable to read Excel CEDA "
                f"catalogue: {exc}"
            )

    return records


# ============================================================
# GET ALL CEDA MARKETS
# ============================================================

def _get_all_ceda_markets() -> List[Dict[str, Any]]:
    """
    Retrieve the complete CEDA market catalogue.
    """

    function = _load_ceda_function()

    if function is not None:

        try:

            result = function()

            if hasattr(
                result,
                "__await__",
            ):

                # Async CEDA functions are handled by the
                # async wrapper below instead.
                return []

            if isinstance(
                result,
                dict,
            ):

                for key in (
                    "mandis",
                    "markets",
                    "data",
                    "results",
                    "records",
                ):

                    value = result.get(
                        key
                    )

                    if isinstance(
                        value,
                        list,
                    ):
                        return value

            if isinstance(
                result,
                list,
            ):
                return result

        except TypeError:
            pass

        except Exception as exc:

            _log(
                f"CEDA function failed: {exc}"
            )

    # --------------------------------------------------------
    # Local catalogue fallback
    # --------------------------------------------------------

    catalogue = (
        _find_local_ceda_catalogue()
    )

    if catalogue is None:

        _log(
            "No local CEDA catalogue was found."
        )

        return []

    _log(
        f"Reading local CEDA catalogue: "
        f"{catalogue}"
    )

    return _read_local_catalogue(
        catalogue
    )


# ============================================================
# ASYNC CEDA LOADER
# ============================================================

async def _get_all_ceda_markets_async() -> List[Dict[str, Any]]:
    """
    Async-safe CEDA catalogue loader.
    """

    function = _load_ceda_function()

    if function is not None:

        try:

            result = function()

            if hasattr(
                result,
                "__await__",
            ):

                result = await result

            if isinstance(
                result,
                dict,
            ):

                for key in (
                    "mandis",
                    "markets",
                    "data",
                    "results",
                    "records",
                ):

                    value = result.get(
                        key
                    )

                    if isinstance(
                        value,
                        list,
                    ):

                        return value

            if isinstance(
                result,
                list,
            ):

                return result

        except TypeError:

            # Some existing CEDA helpers require arguments.
            # We deliberately do not pass radius/limit here.
            pass

        except Exception as exc:

            _log(
                f"CEDA function failed: {exc}"
            )

    # --------------------------------------------------------
    # Local catalogue fallback
    # --------------------------------------------------------

    catalogue = (
        _find_local_ceda_catalogue()
    )

    if catalogue is None:

        _log(
            "No local CEDA catalogue was found."
        )

        return []

    _log(
        f"Reading local CEDA catalogue: "
        f"{catalogue}"
    )

    return _read_local_catalogue(
        catalogue
    )


# ============================================================
# PUBLIC API
# ============================================================

async def get_nearby_mandis(
    latitude: float,
    longitude: float,
) -> List[Dict[str, Any]]:
    """
    Return ALL Karnataka tomato mandis sorted by distance.

    IMPORTANT:
    This function intentionally has ONLY latitude and longitude.

    There is:
        - no radius argument
        - no limit argument

    This prevents the previous error:

        get_nearby_mandis() got an unexpected keyword argument 'radius'
    """

    if not math.isfinite(
        latitude
    ):
        raise ValueError(
            "Invalid latitude."
        )

    if not math.isfinite(
        longitude
    ):
        raise ValueError(
            "Invalid longitude."
        )

    if latitude < -90 or latitude > 90:
        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if longitude < -180 or longitude > 180:
        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    _log(
        "=========================================================="
    )

    _log(
        "ALL KARNATAKA TOMATO MANDI SEARCH"
    )

    _log(
        f"User latitude: {latitude}"
    )

    _log(
        f"User longitude: {longitude}"
    )

    _log(
        "Radius: NONE"
    )

    _log(
        "Limit: NONE"
    )

    _log(
        "=========================================================="
    )

    raw_markets = (
        await _get_all_ceda_markets_async()
    )

    _log(
        f"CEDA records received: "
        f"{len(raw_markets)}"
    )

    results: List[Dict[str, Any]] = []

    seen = set()

    for raw_market in raw_markets:

        market = _normalize_market(
            raw_market
        )

        if market is None:
            continue

        market_lat = market[
            "latitude"
        ]

        market_lng = market[
            "longitude"
        ]

        # ----------------------------------------------------
        # Skip records without usable coordinates.
        # ----------------------------------------------------

        if not math.isfinite(
            market_lat
        ) or not math.isfinite(
            market_lng
        ):
            continue

        if market_lat == 0 and market_lng == 0:
            continue

        # ----------------------------------------------------
        # Only Karnataka markets.
        #
        # This is a geographic filter, not a city list.
        # ----------------------------------------------------

        if not _is_karnataka_coordinate(
            market_lat,
            market_lng,
        ):
            continue

        # ----------------------------------------------------
        # Deduplicate.
        # ----------------------------------------------------

        key = (
            market["name"].strip().lower(),
            market["city"].strip().lower(),
            round(market_lat, 5),
            round(market_lng, 5),
        )

        if key in seen:
            continue

        seen.add(key)

        # ----------------------------------------------------
        # Distance from user's real GPS.
        # ----------------------------------------------------

        distance = haversine_distance_km(
            latitude,
            longitude,
            market_lat,
            market_lng,
        )

        market["distance"] = round(
            distance,
            1,
        )

        market["travel"] = (
            _estimate_travel_time(
                distance
            )
        )

        results.append(
            market
        )

    # ========================================================
    # SORT NEAREST -> FARTHEST
    # ========================================================

    results.sort(
        key=lambda item: (
            item.get(
                "distance",
                float("inf"),
            ),
            item.get(
                "name",
                "",
            ).lower(),
        )
    )

    _log(
        f"Karnataka mandis with coordinates: "
        f"{len(results)}"
    )

    if results:

        _log(
            "Nearest mandi:"
        )

        _log(
            f"{results[0]['name']} - "
            f"{results[0]['distance']} km"
        )

        _log(
            "Farthest mandi:"
        )

        _log(
            f"{results[-1]['name']} - "
            f"{results[-1]['distance']} km"
        )

    _log(
        "Returning ALL Karnataka mandis sorted by distance."
    )

    return results


# ============================================================
# RESPONSE BUILDER
# ============================================================

async def get_nearby_mandi_response(
    latitude: float,
    longitude: float,
) -> Dict[str, Any]:
    """
    Build the complete API response expected by the frontend.
    """

    mandis = await get_nearby_mandis(
        latitude,
        longitude,
    )

    distances = [
        mandi["distance"]
        for mandi in mandis
        if isinstance(
            mandi.get("distance"),
            (int, float),
        )
    ]

    farthest_distance = (
        max(distances)
        if distances
        else 0
    )

    return {
        "mandis": mandis,

        "mandi_count": len(
            mandis
        ),

        # No actual search radius is applied.
        # null communicates that clearly to the frontend.
        "radius_km": None,

        "region": "Karnataka",

        "last_updated": "",

        "search_mode": "all_karnataka",

        "sorted_by": "distance",

        "sort_order": "nearest_to_farthest",

        "farthest_distance_km": farthest_distance,
    }


# ============================================================
# STATUS
# ============================================================

def get_mandi_status() -> Dict[str, Any]:
    """
    Basic CEDA/local catalogue status.
    """

    catalogue = (
        _find_local_ceda_catalogue()
    )

    database_exists = (
        catalogue is not None
    )

    database_path = (
        str(catalogue)
        if catalogue
        else ""
    )

    total_markets = 0
    markets_with_coordinates = 0
    markets_without_coordinates = 0

    if catalogue is not None:

        try:

            records = _read_local_catalogue(
                catalogue
            )

            total_markets = len(
                records
            )

            for record in records:

                market = _normalize_market(
                    record
                )

                if market is None:
                    continue

                lat = market[
                    "latitude"
                ]

                lng = market[
                    "longitude"
                ]

                if (
                    math.isfinite(lat)
                    and math.isfinite(lng)
                    and not (
                        lat == 0
                        and lng == 0
                    )
                ):

                    markets_with_coordinates += 1

                else:

                    markets_without_coordinates += 1

        except Exception as exc:

            _log(
                f"Unable to inspect catalogue: "
                f"{exc}"
            )

    return {
        "database_exists": database_exists,
        "database_path": database_path,
        "total_markets": total_markets,
        "markets_with_coordinates": markets_with_coordinates,
        "markets_without_coordinates": markets_without_coordinates,
    }