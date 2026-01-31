import asyncio
import httpx
import time
import statistics

# Configuration
BASE_URL = "http://localhost:8000"
QUERY_TEXT = "Tell me about applied devops class"
CONCURRENT_REQUESTS = 10

async def call_endpoint(client, endpoint_path, label):
    start_time = time.perf_counter()
    try:
        # Construct full URL correctly
        url = f"{BASE_URL}{endpoint_path}?text={QUERY_TEXT}"
        response = await client.post(url, timeout=30.0)
        response.raise_for_status()
        end_time = time.perf_counter()
        return end_time - start_time
    except Exception as e:
        print(f"Error calling {label}: {e}")
        return None

async def run_load_test(endpoint_path, label):
    print(f"\n--- Starting Load Test: {label} ({endpoint_path}) ---")
    
    async with httpx.AsyncClient() as client:
        # Warm up (optional but recommended)
        await call_endpoint(client, endpoint_path, label)
        
        # Run concurrent requests
        tasks = [call_endpoint(client, endpoint_path, label) for _ in range(CONCURRENT_REQUESTS)]
        
        start_total = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_total = time.perf_counter()
        
        # Filter out failed requests
        latencies = [r for r in results if r is not None]
        
        if latencies:
            print(f"Successful requests: {len(latencies)}/{CONCURRENT_REQUESTS}")
            print(f"Total time for {CONCURRENT_REQUESTS} requests: {end_total - start_total:.4f}s")
            print(f"Average Latency: {statistics.mean(latencies):.4f}s")
            print(f"Median Latency: {statistics.median(latencies):.4f}s")
            print(f"Min Latency: {min(latencies):.4f}s")
            print(f"Max Latency: {max(latencies):.4f}s")
        else:
            print("All requests failed.")

async def main():
    print(f"Load Test Configuration:")
    print(f"Concurrent Queries: {CONCURRENT_REQUESTS}")
    print(f"Query: '{QUERY_TEXT}'")
    
    # Test Sync Endpoint
    # Note: Assumes you run 'uvicorn app.app:app' on port 8000
    await run_load_test("/query", "Sync Endpoint")
    
    print("\n" + "="*50)
    print("Switch to the other terminal and run the async app now if they are on the same port,")
    print("or update the BASE_URL in this script to test the other one.")
    print("="*50)
    
    # Test Async Endpoint
    # Note: Assumes you run 'uvicorn app.app2:app' on port 8000
    # await run_load_test("/query", "Async Endpoint")

if __name__ == "__main__":
    asyncio.run(main())

