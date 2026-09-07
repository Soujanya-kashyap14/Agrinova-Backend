"""
CEDA Agri Market (Agmarknet) client.

Uses the local commodities/geographies catalogue shipped in data/mandi,
and the live CEDA API for market lists and tomato prices when a key is set.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from config import get_settings


CEDA_BASE_URL = "https://api.ceda.ashoka.edu.in/v1"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mandi"
TOMATO_COMMODITY_ID = 78

_session: Optional[requests.Session] = None
_commodities: Optional[List[Dict[str, Any]]] = None
_geographies: Optional[List[Dict[str, Any]]] = None
_market_cache: Dict[tuple, List[Dict[str, Any]]] = {}
_price_cache: Dict[tuple, List[Dict[str, Any]]] = {}


def _log(message: str) -> None:
    print(f"[CEDA] {message}")


def _get_session() -> requests.Session:
    global _session

    if _session is None:
        settings = get_settings()
        session = requests.Session()
        api_key = (settings.ceda_api_key or "").strip()
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        session.headers["Accept"] = "application/json"
        _session = session

    return _session


def has_api_key() -> bool:
    return bool((get_settings().ceda_api_key or "").strip())


def _unwrap(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    output = payload.get("output")
    if isinstance(output, dict):
        data = output.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

    for key in ("data", "records", "markets", "mandis", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def _request(
    method: str,
    path: str,
    json_body: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    url = f"{CEDA_BASE_URL}{path}"
    session = _get_session()

    for attempt in range(max_retries + 1):
        try:
            response = session.request(
                method,
                url,
                json=json_body,
                timeout=30,
            )
        except requests.RequestException as exc:
            _log(f"Network error {method} {path}: {exc}")
            if attempt >= max_retries:
                return []
            time.sleep(1.2 * (attempt + 1))
            continue

        if response.status_code == 429 and attempt < max_retries:
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 1.5 * (attempt + 1)
            time.sleep(delay)
            continue

        if response.status_code in (401, 403):
            _log("CEDA API rejected the request. Check CEDA_API_KEY.")
            return []

        if not response.ok:
            _log(f"HTTP {response.status_code} for {method} {path}")
            return []

        try:
            return _unwrap(response.json())
        except ValueError:
            _log(f"Invalid JSON from {path}")
            return []

    return []


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    return _unwrap(data)


def list_commodities() -> List[Dict[str, Any]]:
    global _commodities

    if _commodities is not None:
        return _commodities

    local = _load_json_list(DATA_DIR / "commodities.json")
    if local:
        _commodities = local
        return _commodities

    remote = _request("GET", "/agmarknet/commodities")
    _commodities = remote
    return _commodities


def list_geographies() -> List[Dict[str, Any]]:
    global _geographies

    if _geographies is not None:
        return _geographies

    local = _load_json_list(DATA_DIR / "geographies.json")
    if local:
        _geographies = local
        return _geographies

    remote = _request("GET", "/agmarknet/geographies")
    _geographies = remote
    return _geographies


def tomato_commodity_id() -> int:
    for item in list_commodities():
        name = str(item.get("commodity_name") or "").strip().lower()
        if name == "tomato":
            try:
                return int(item.get("commodity_id"))
            except (TypeError, ValueError):
                break
    return TOMATO_COMMODITY_ID


def list_markets(
    commodity_id: int,
    state_id: int,
    district_id: int,
    indicator: str = "price",
) -> List[Dict[str, Any]]:
    key = (commodity_id, state_id, district_id, indicator)
    if key in _market_cache:
        return _market_cache[key]

    if not has_api_key():
        _market_cache[key] = []
        return []

    data = _request(
        "POST",
        "/agmarknet/markets",
        {
            "commodity_id": commodity_id,
            "state_id": state_id,
            "district_id": district_id,
            "indicator": indicator,
        },
    )
    _market_cache[key] = data
    return data


def list_prices(
    commodity_id: int,
    state_id: int,
    district_ids: Optional[List[int]] = None,
    lookback_days: int = 400,
) -> List[Dict[str, Any]]:
    district_key = tuple(sorted(district_ids or []))
    key = (commodity_id, state_id, district_key, lookback_days)
    if key in _price_cache:
        return _price_cache[key]

    if not has_api_key():
        _price_cache[key] = []
        return []

    today = date.today()
    body: Dict[str, Any] = {
        "commodity_id": commodity_id,
        "state_id": state_id,
        "from_date": (today - timedelta(days=lookback_days)).isoformat(),
        "to_date": today.isoformat(),
    }
    if district_ids:
        body["district_id"] = district_ids

    data = _request("POST", "/agmarknet/prices", body)
    _price_cache[key] = data
    return data


def latest_prices_by_market(
    commodity_id: int,
    state_id: int,
    district_ids: Optional[List[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    rows = list_prices(commodity_id, state_id, district_ids)
    latest: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        market_id = row.get("market_id")
        if market_id is None:
            continue

        try:
            market_key = int(market_id)
        except (TypeError, ValueError):
            continue

        current = latest.get(market_key)
        if current is None or str(row.get("date") or "") > str(current.get("date") or ""):
            latest[market_key] = row

    return latest
