import osmnx as ox
import geopandas as gpd

north, south, east, west = -1.1273563, 37.0312786, -1.124858, 37.0338659  # ← your bbox from Day 2

buildings = ox.features_from_bbox(
    bbox=(north, south, east, west),
    tags={"building": True}
)
print(f"Found {len(buildings)} buildings")
buildings.plot(figsize=(10, 10))