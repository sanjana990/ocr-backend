#!/usr/bin/env python3
"""
Test script for the /crawl-company endpoint
"""

import requests
import json
import time

def test_crawl_endpoint():
    """Test the crawl-company endpoint"""
    
    # Server URL
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        {
            "name": "Microsoft LinkedIn",
            "params": {
                "company_name": "microsoft",
                "use_ai_extraction": True,
                "platform": "linkedin"
            }
        },
        {
            "name": "Google LinkedIn", 
            "params": {
                "company_name": "google",
                "use_ai_extraction": True,
                "platform": "linkedin"
            }
        },
        {
            "name": "Apple LinkedIn",
            "params": {
                "company_name": "apple",
                "use_ai_extraction": True,
                "platform": "linkedin"
            }
        }
    ]
    
    print("🚀 Testing /crawl-company endpoint...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            # Make the request
            response = requests.post(
                f"{base_url}/crawl-company",
                params=test_case['params'],
                timeout=60  # 60 second timeout
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {data.get('success', False)}")
                print(f"Company: {data.get('company_name', 'N/A')}")
                print(f"URL: {data.get('url', 'N/A')}")
                print(f"Crawl Time: {data.get('crawl_time', 0):.2f}s")
                print(f"Content Length: {data.get('content_length', 0)}")
                print(f"AI Extraction: {data.get('ai_extraction', {}).get('extraction_successful', False)}")
                print(f"Saved to DB: {data.get('saved_to_database', False)}")
                
                # Show AI extraction preview
                ai_data = data.get('ai_extraction', {})
                if ai_data.get('raw_ai_response'):
                    print(f"AI Response Preview: {ai_data['raw_ai_response'][:200]}...")
                    
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out (60s)")
        except requests.exceptions.ConnectionError:
            print("🔌 Connection error - is the server running?")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Add delay between requests to respect rate limits
        if i < len(test_cases):
            print("⏸️ Waiting 5 seconds before next request...")
            time.sleep(5)
    
    print("\n" + "=" * 50)
    print("✅ Testing completed!")

def test_batch_endpoint():
    """Test the /crawl-multiple-companies endpoint"""
    
    print("\n🚀 Testing /crawl-multiple-companies endpoint...")
    print("=" * 50)
    
    try:
        response = requests.post(
            "http://localhost:8000/crawl-multiple-companies",
            params={
                "company_names": ["microsoft", "google"],
                "use_ai_extraction": True,
                "platform": "linkedin"
            },
            timeout=120  # 2 minute timeout for batch
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data.get('success', False)}")
            print(f"Total Companies: {data.get('total_companies', 0)}")
            print(f"Successful Crawls: {data.get('successful_crawls', 0)}")
            
            for i, result in enumerate(data.get('results', []), 1):
                print(f"\nCompany {i}: {result.get('company_name', 'N/A')}")
                print(f"  Success: {result.get('success', False)}")
                if result.get('success'):
                    print(f"  Crawl Time: {result.get('crawl_time', 0):.2f}s")
                    print(f"  Content Length: {result.get('content_length', 0)}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown')}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Crawl4AI Endpoint Tester")
    print("Make sure your server is running on http://localhost:8000")
    print("Press Enter to start testing...")
    input()
    
    # Test single company endpoint
    test_crawl_endpoint()
    
    # Test batch endpoint
    test_batch_endpoint()
