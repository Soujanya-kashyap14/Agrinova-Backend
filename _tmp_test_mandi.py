import asyncio

from services.mandi_service import get_nearby_mandi_response


async def main() -> None:
    result = await get_nearby_mandi_response(location="Mysore", sort_by="distance")
    print("count", result["mandi_count"])
    print("label", result["location_label"])
    print("sorted_by", result["sorted_by"])
    print("ceda_live", result["ceda_live"])
    print("--- nearest ---")
    for mandi in result["mandis"][:8]:
        print(
            f"{mandi['distance']:6.1f} km  r={mandi['rating']:.1f}  "
            f"{mandi['name']} ({mandi['city']})"
        )

    ranked = await get_nearby_mandi_response(location="Mysore", sort_by="rating")
    print("--- rating ---")
    for mandi in ranked["mandis"][:8]:
        print(
            f"{mandi['rating']:.1f}  {mandi['distance']:6.1f} km  {mandi['name']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
