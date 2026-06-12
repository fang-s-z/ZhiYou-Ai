import requests
import time

print("Testing health endpoint...")
try:
    r = requests.get('http://localhost:8000/api/health', timeout=10)
    print(f"Health check: {r.status_code} - {r.text}")
except Exception as e:
    print(f"Health check failed: {e}")

print("\nTesting generate-plan endpoint...")
start_time = time.time()
try:
    r = requests.post('http://localhost:8000/api/generate-plan', 
                      json={'destination':'厦门','days':2,'budget':'舒适','people_count':2}, 
                      timeout=120)
    elapsed = time.time() - start_time
    print(f"Status: {r.status_code}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Response: {r.text[:2000]}")
except requests.exceptions.Timeout:
    print(f"Timeout after {time.time() - start_time:.2f}s")
except Exception as e:
    print(f"Error: {type(e).__name__} - {e}")