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


def offset_metres(
    point: tuple[float, float],
    metres: float,
    bearing_radians: float,
) -> tuple[float, float]:
    """Move a (latitude, longitude) point `metres` along a bearing from north.

    A local flat-earth step, exact to well under a millimetre at the few hundred
    metres the reviewer checks use (section 7.15), and invertible by
    `haversine_metres` to the same precision.
    """
    latitude = math.radians(point[0])
    north = metres * math.cos(bearing_radians)
    east = metres * math.sin(bearing_radians)
    return (
        point[0] + math.degrees(north / _EARTH_RADIUS_METRES),
        point[1] + math.degrees(east / (_EARTH_RADIUS_METRES * math.cos(latitude))),
    )
