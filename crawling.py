import os; 
import asyncio
from dotenv import load_dotenv
import time
from collections import deque
from crawl4ai import AsyncWebCrawler
from google import genai 


load_dotenv()

class SimpleRateLimiter:
    """Simple rate limiter - 3 requests per 60 seconds with 2 second delays"""
    
    def __init__(self):
        self.requests = deque()  # Track request times
        self.last_request = 0
        self.max_requests = 3    # Max 3 requests per minute
        self.time_window = 60    # 60 seconds
        self.min_delay = 2       # 2 seconds between requests
    
    async def wait_if_needed(self):
        """Wait if we need to respect rate limits"""
        now = time.time()
        
        # Remove old requests (older than 60 seconds)
        while self.requests and now - self.requests[0] > self.time_window:
            self.requests.popleft()
        
        # If we've hit the limit, wait until the oldest request expires
        if len(self.requests) >= self.max_requests:
            wait_time = self.time_window - (now - self.requests[0]) + 1
            print(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
            now = time.time()
        # Wait minimum delay between requests
        if self.last_request > 0:
            time_since_last = now - self.last_request
            if time_since_last < self.min_delay:
                delay = self.min_delay - time_since_last
                print(f"⏸️ Waiting {delay:.1f}s between requests...")
                await asyncio.sleep(delay)
                now = time.time()
        
        # Record this request
        self.requests.append(now)
        self.last_request = now
        print(f"✅ Request approved ({len(self.requests)}/{self.max_requests} requests used)")

browser_config = {
    "browser": "Chrome",
    "window_size": {"width": 1980, "height": 1080},
    "headless": True,
    "verbose": True,
    "enable_stealth": True,
    "enable_logging": True,
}

rate_limiter = SimpleRateLimiter()

gemini_key = os.environ["gemini_api_key"]

class CompanyInfo: 
    def __init__(self):
        self.company = ""
        self.description = ""
        self.products_services = []
        self.industry = ""
        self.headquarters = ""
        self.founding_date = ""
        self.key_people = []
        self.revenue = []
        self.number_of_employees = []

# Use GPT-4.1 nano for cost-effective extraction
strategy = {
    "instruction":"""Extract the following company information:
    - Company name
    - Description/tagline
    - Products/services
    - Location/headquarters
    - Industry
    - Number of employees
    Return as JSON."""
}

company = "jiohotstar"

async def main():
    await rate_limiter.wait_if_needed()
    print(f"🔍 Crawling: {company}")
    start_time = time.time()

    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        # Run the crawler on a URL
        result = await crawler.arun(url=f"https://www.linkedin.com/company/{company}/")

        # Print the extracted content
        elapsed = time.time() - start_time
        print(f"⏱️ Crawling completed in {elapsed:.1f} seconds")
        if not result.success:
            print("❌ Crawl unsuccessful")
            return None
        else:
            client = genai.Client(api_key=os.getenv("gemini_api_key"))

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{strategy} {result.markdown}"
            )

            return response.text

# Run the async main function
res = asyncio.run(main())
print(res)

