#!/usr/bin/env python3
"""
Simple test script to verify the endpoint is working before running the full generation.
"""

import requests
import json

def test_endpoint(endpoint_url: str):
    """Test the endpoint with a sample game recap"""
    
    # Sample game recap for testing
    sample_recap = """
    CHARLOTTE, N.C. -- Houston Rockets coach Mike D'Antoni seemed genuinely disappointed his team only attempted 55 3-pointers against the Charlotte Hornets.
    "I don't know why we didn't shoot 60," D'Antoni quipped. "We passed up about four."
    James Harden had his first triple-double of the season with 27 points, 11 assists and 10 rebounds, and the Rockets finished 22 of 57 from beyond the 3-point arc in a 109-93 win Friday night.
    Houston nearly broke its own NBA regular season records of 24 3-pointers and 61 3-pointers attempted set last December.
    Eric Gordon and Ryan Anderson led the 3-point barrage. Gordon was 6 of 16 from beyond the arc and had 26 points, while Anderson was 6 of 15 and finished with 21 points.
    """
    
    payload = {
        "game_recap": sample_recap.strip(),
        "max_length": 100
    }
    
    print(f"Testing endpoint: {endpoint_url}")
    print("Sample game recap:")
    print(sample_recap.strip())
    print("\n" + "="*50)
    
    try:
        # Test health endpoint first
        print("1. Testing health endpoint...")
        health_response = requests.get(f"{endpoint_url}/health", timeout=10)
        health_response.raise_for_status()
        print("✅ Health check passed")
        
        # Test inference endpoint
        print("2. Testing inference endpoint...")
        response = requests.post(
            f"{endpoint_url}/summarize_recap",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        generated_summary = result.get("game_recap_summary", "")
        
        print("✅ Inference test passed")
        print(f"Generated summary: {generated_summary}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    endpoint_url = "http://54.197.213.231:8000"
    success = test_endpoint(endpoint_url)
    
    if success:
        print("\n🎉 Endpoint is working! You can now run the full generation script.")
    else:
        print("\n❌ Endpoint test failed. Please check the EC2 instance status.")
