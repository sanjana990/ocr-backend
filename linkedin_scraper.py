#!/usr/bin/env python3
"""
LinkedIn Company Scraper - Integrated with existing backend
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import scrapy
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    from scrapy import Spider
    from scrapy_playwright.page import PageMethod
    SCRAPY_AVAILABLE = True
except ImportError as e:
    SCRAPY_AVAILABLE = False
    print(f"⚠️ Scrapy not available: {e}")
    print("Install with: pip install scrapy scrapy-playwright")

class LinkedInCompanySpider(Spider):
    name = "linkedin_company_profile"
    
    def __init__(self, company_name=None, company_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.company_name = company_name
        self.company_url = company_url
        self.start_urls = [company_url] if company_url else []
        
    custom_settings = {
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "ROBOTSTXT_OBEY": False,
        "DOWNLOAD_DELAY": 2,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
    }

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                meta=dict(
                    playwright=True,
                    playwright_include_page=True,
                    playwright_page_methods=[
                        PageMethod("wait_for_timeout", 3000),
                        PageMethod("wait_for_selector", "h1", timeout=10000),
                    ],
                ),
                callback=self.parse,
            )

    def parse(self, response):
        self.logger.info(f"Scraping LinkedIn company: {response.url}")
        
        # Extract company information
        company_name = response.css('h1::text').get()
        if company_name:
            company_name = company_name.strip()
        
        # Extract structured data using dt/dd pattern
        website = self._extract_field(response, "Website", "a::attr(href)")
        industry = self._extract_field(response, "Industry", "::text")
        size = self._extract_field(response, "Company size", "::text")
        hq_location = self._extract_field(response, "Headquarters", "::text")
        company_type = self._extract_field(response, "Type", "::text")
        
        data = {
            "company_name": company_name,
            "website": website,
            "industry": industry,
            "size": size,
            "hq_location": hq_location,
            "company_type": company_type,
            "url": response.url,
            "status": response.status,
            "scraped_at": self._get_timestamp()
        }
        
        self.logger.info(f"Extracted data: {data}")
        yield data

    def _extract_field(self, response, field_name, selector):
        """Extract field using dt:contains pattern"""
        try:
            dd_element = response.css(f'dt:contains("{field_name}") + dd')
            if dd_element:
                if selector == "a::attr(href)":
                    return dd_element.css('a::attr(href)').get()
                else:
                    return dd_element.css(selector).get()
        except Exception as e:
            self.logger.warning(f"Failed to extract {field_name}: {e}")
        return None

    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


class LinkedInScraperService:
    """Service class for LinkedIn company scraping"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def scrape_company(self, company_name: str) -> Dict[str, Any]:
        """
        Scrape LinkedIn company information
        
        Args:
            company_name: Name of the company to scrape
            
        Returns:
            Dict containing company information
        """
        if not SCRAPY_AVAILABLE:
            raise ImportError("Scrapy not available. Install with: pip install scrapy scrapy-playwright")
        
        # Construct LinkedIn company URL
        company_slug = company_name.lower().replace(" ", "-").replace("&", "and")
        company_url = f"https://www.linkedin.com/company/{company_slug}/"
        
        self.logger.info(f"Scraping LinkedIn for company: {company_name}")
        self.logger.info(f"Company URL: {company_url}")
        
        # For now, return mock data to test the endpoint
        # TODO: Implement actual scraping
        mock_data = {
            "company_name": company_name,
            "website": f"https://{company_slug}.com",
            "industry": "Technology",
            "size": "10,001+ employees",
            "hq_location": "Mountain View, CA",
            "company_type": "Public Company",
            "url": company_url,
            "scraped_at": self._get_timestamp()
        }
        
        self.logger.info(f"Mock data generated for: {company_name}")
        return mock_data
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def scrape_multiple_companies(self, company_names: list) -> list:
        """
        Scrape multiple companies
        
        Args:
            company_names: List of company names to scrape
            
        Returns:
            List of company information dictionaries
        """
        results = []
        for company_name in company_names:
            try:
                result = await self.scrape_company(company_name)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to scrape {company_name}: {e}")
                results.append({"error": str(e), "company_name": company_name})
        
        return results


# Standalone usage
if __name__ == "__main__":
    import asyncio
    
    async def test_scraper():
        scraper = LinkedInScraperService()
        
        # Test with a single company
        result = await scraper.scrape_company("Google")
        print("Scraping result:", json.dumps(result, indent=2))
    
    # Run the test
    asyncio.run(test_scraper())
