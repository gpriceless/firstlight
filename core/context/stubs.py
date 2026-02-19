"""
Synthetic data generators for Context Data Lakehouse.

Since the pipeline has no existing fetch paths for building footprints,
infrastructure facilities, or weather observations, these stub generators
produce realistic synthetic data for a given bounding box. Each generator
returns a list of the corresponding Pydantic model ready for
ContextRepository.store_*() or store_batch().

All synthetic records use source="synthetic" so they can be distinguished
from real data when real OSM/Overture/NOAA integrations are added later.
"""

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.context.models import BuildingRecord, InfrastructureRecord, WeatherRecord


# ---------------------------------------------------------------------------
# Building footprint generator
# ---------------------------------------------------------------------------

def generate_buildings(
    bbox: Tuple[float, float, float, float],
    count: int = 50,
    seed: Optional[int] = None,
) -> List[BuildingRecord]:
    """
    Generate synthetic building footprints inside *bbox*.

    Each building is a small rectangular polygon with random offsets
    to simulate realistic footprints.

    Args:
        bbox: (west, south, east, north) in WGS84.
        count: Number of buildings to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of BuildingRecord with source="synthetic".
    """
    rng = random.Random(seed)
    west, south, east, north = bbox
    lon_range = east - west
    lat_range = north - south

    buildings: List[BuildingRecord] = []

    building_types = [
        "residential", "commercial", "industrial", "retail",
        "office", "warehouse", "school", "church",
    ]

    for _ in range(count):
        # Random center point inside bbox
        cx = west + rng.random() * lon_range
        cy = south + rng.random() * lat_range

        # Small rectangle (~20-80m across), random orientation skew
        half_w = rng.uniform(0.0001, 0.0004)  # ~10-40m in longitude
        half_h = rng.uniform(0.0001, 0.0003)  # ~10-30m in latitude
        skew = rng.uniform(-0.00005, 0.00005)

        coords = [
            [cx - half_w + skew, cy - half_h],
            [cx + half_w + skew, cy - half_h],
            [cx + half_w - skew, cy + half_h],
            [cx - half_w - skew, cy + half_h],
            [cx - half_w + skew, cy - half_h],  # close ring
        ]

        building_type = rng.choice(building_types)
        floors = rng.randint(1, 5) if building_type != "warehouse" else 1

        buildings.append(
            BuildingRecord(
                source="synthetic",
                source_id=f"synth_bldg_{uuid.uuid4().hex[:12]}",
                geometry={"type": "Polygon", "coordinates": [coords]},
                properties={
                    "type": building_type,
                    "floors": floors,
                    "height_m": floors * rng.uniform(2.8, 3.5),
                    "area_sqm": round(half_w * 2 * 111_000 * half_h * 2 * 111_000, 1),
                },
            )
        )

    return buildings


# ---------------------------------------------------------------------------
# Infrastructure facility generator
# ---------------------------------------------------------------------------

# Facility templates: (type, name_prefix, geometry_type, typical_capacity)
_INFRASTRUCTURE_TEMPLATES = [
    ("hospital", "Regional Hospital", "POLYGON", 200),
    ("hospital", "Community Hospital", "POINT", 80),
    ("fire_station", "Fire Station", "POINT", 15),
    ("police_station", "Police Station", "POINT", 25),
    ("school", "Public School", "POLYGON", 500),
    ("power_plant", "Power Substation", "POINT", 0),
    ("water_treatment", "Water Treatment Facility", "POLYGON", 0),
    ("emergency_shelter", "Emergency Shelter", "POINT", 300),
]


def generate_infrastructure(
    bbox: Tuple[float, float, float, float],
    count: int = 10,
    seed: Optional[int] = None,
) -> List[InfrastructureRecord]:
    """
    Generate synthetic infrastructure facilities inside *bbox*.

    Produces a mix of POINT (hospitals, fire stations) and POLYGON
    (hospital campuses, school campuses) geometries to exercise the
    GEOMETRY(GEOMETRY, 4326) mixed-type column.

    Args:
        bbox: (west, south, east, north) in WGS84.
        count: Number of facilities to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of InfrastructureRecord with source="synthetic".
    """
    rng = random.Random(seed)
    west, south, east, north = bbox
    lon_range = east - west
    lat_range = north - south

    facilities: List[InfrastructureRecord] = []

    for i in range(count):
        template = _INFRASTRUCTURE_TEMPLATES[i % len(_INFRASTRUCTURE_TEMPLATES)]
        ftype, name_prefix, geom_type, capacity = template

        cx = west + rng.random() * lon_range
        cy = south + rng.random() * lat_range

        if geom_type == "POLYGON":
            # Campus-sized polygon (~100-300m)
            half_w = rng.uniform(0.0005, 0.0015)
            half_h = rng.uniform(0.0005, 0.0015)
            geometry: Dict[str, Any] = {
                "type": "Polygon",
                "coordinates": [[
                    [cx - half_w, cy - half_h],
                    [cx + half_w, cy - half_h],
                    [cx + half_w, cy + half_h],
                    [cx - half_w, cy + half_h],
                    [cx - half_w, cy - half_h],
                ]],
            }
        else:
            geometry = {
                "type": "Point",
                "coordinates": [cx, cy],
            }

        facilities.append(
            InfrastructureRecord(
                source="synthetic",
                source_id=f"synth_infra_{uuid.uuid4().hex[:12]}",
                geometry=geometry,
                properties={
                    "type": ftype,
                    "name": f"{name_prefix} #{i + 1}",
                    "capacity": capacity + rng.randint(-20, 50) if capacity else None,
                    "operational": True,
                },
            )
        )

    return facilities


# ---------------------------------------------------------------------------
# Weather observation generator
# ---------------------------------------------------------------------------


def generate_weather(
    bbox: Tuple[float, float, float, float],
    count: int = 10,
    reference_time: Optional[datetime] = None,
    seed: Optional[int] = None,
) -> List[WeatherRecord]:
    """
    Generate synthetic weather observations inside *bbox*.

    Each observation is a POINT with realistic temperature, precipitation,
    and wind values.

    Args:
        bbox: (west, south, east, north) in WGS84.
        count: Number of observations to generate.
        reference_time: Base time for observations (defaults to now).
        seed: Random seed for reproducibility.

    Returns:
        List of WeatherRecord with source="synthetic".
    """
    rng = random.Random(seed)
    west, south, east, north = bbox
    lon_range = east - west
    lat_range = north - south

    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    observations: List[WeatherRecord] = []

    for i in range(count):
        cx = west + rng.random() * lon_range
        cy = south + rng.random() * lat_range

        # Observation time within the last 24 hours relative to reference
        hours_offset = rng.uniform(0, 24)
        obs_time = reference_time - timedelta(hours=hours_offset)

        # Realistic weather values
        temp_c = rng.uniform(-5.0, 40.0)
        precip_mm = max(0.0, rng.gauss(2.0, 5.0))
        wind_speed_ms = max(0.0, rng.gauss(5.0, 3.0))
        wind_dir_deg = rng.uniform(0, 360)
        humidity_pct = rng.uniform(20.0, 100.0)
        pressure_hpa = rng.uniform(990.0, 1030.0)

        observations.append(
            WeatherRecord(
                source="synthetic",
                source_id=f"synth_wx_{uuid.uuid4().hex[:12]}",
                geometry={
                    "type": "Point",
                    "coordinates": [round(cx, 6), round(cy, 6)],
                },
                properties={
                    "temperature_c": round(temp_c, 1),
                    "precipitation_mm": round(precip_mm, 1),
                    "wind_speed_ms": round(wind_speed_ms, 1),
                    "wind_direction_deg": round(wind_dir_deg, 0),
                    "humidity_pct": round(humidity_pct, 1),
                    "pressure_hpa": round(pressure_hpa, 1),
                },
                observation_time=obs_time,
            )
        )

    return observations
