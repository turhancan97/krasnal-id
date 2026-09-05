"""Great-circle distance, shared by the data pipeline and the experiments."""

import math

_EARTH_RADIUS_METRES = 6371008.8


def haversine_metres(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Great-circle distance between two (latitude, longitude) pairs, in metres."""
    lat1, lon1 = math.radians(first[0]), math.radians(first[1])
    lat2, lon2 = math.radians(second[0]), math.radians(second[1])
    delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
    inner = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_METRES * math.asin(math.sqrt(min(1.0, inner)))
