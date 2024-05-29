import math
import random

def calculate_arc_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    # Earth's radius in kilometers
    R_earth = 6371.0
    
    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Differences in coordinates
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
    central_angle = 2 * math.asin(math.sqrt(a))
    
    # Convert altitude from meters to kilometers
    alt1_km = alt1 / 1000
    alt2_km = alt2 / 1000
    
    # Average altitude
    avg_alt = (alt1_km + alt2_km) / 2
    
    # Calculate the arc distance
    arc_distance = central_angle * (R_earth + avg_alt)
    
    return arc_distance

# Generate random coordinates for the two points
def generate_random_coordinates():
    latitude = random.uniform(-90, 90)  # Latitude between -90 and 90 degrees
    longitude = random.uniform(-180, 180)  # Longitude between -180 and 180 degrees
    altitude = random.uniform(1000, 2000) * 1000  # Altitude between 1000 km and 2000 km above surface
    return latitude, longitude, altitude

# Example usage
lat1, lon1, alt1 = generate_random_coordinates()
lat2, lon2, alt2 = generate_random_coordinates()

distance = calculate_arc_distance(lat1, lon1, alt1, lat2, lon2, alt2)
print(f"Point 1: ({lat1:.2f}°, {lon1:.2f}°) at {alt1/1000:.2f} km altitude")
print(f"Point 2: ({lat2:.2f}°, {lon2:.2f}°) at {alt2/1000:.2f} km altitude")
print(f"The arc distance between the two points is approximately {distance:.2f} kilometers.")
print("Prop delay along arc: " + str(distance*1000/(3e8)))