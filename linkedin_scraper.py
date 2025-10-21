#!/usr/bin/env python3
"""
LinkedIn Company Scraper - Using Scrapfly API for better performance and memory efficiency
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import os
import requests
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from scrapfly import ScrapflyClient, ScrapeConfig
    SCRAPFLY_AVAILABLE = True
except ImportError as e:
    SCRAPFLY_AVAILABLE = False
    print(f"⚠️ Scrapfly not available: {e}")
    print("Install with: pip install scrapfly-sdk")

class LinkedInDataExtractor:
    """Extract LinkedIn company data from HTML content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_company_data(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract company information from LinkedIn HTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract company name
            company_name = self._extract_text(soup, 'h1')
            
            # Extract structured data
            website = self._extract_field(soup, "Website")
            industry = self._extract_field(soup, "Industry")
            size = self._extract_field(soup, "Company size")
            hq_location = self._extract_field(soup, "Headquarters")
            company_type = self._extract_field(soup, "Type")
            
            data = {
                "company_name": company_name,
                "website": website,
                "industry": industry,
                "size": size,
                "hq_location": hq_location,
                "company_type": company_type,
                "url": url,
                "scraped_at": self._get_timestamp()
            }
            
            self.logger.info(f"Extracted data: {data}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to extract data: {e}")
            return {"error": str(e), "url": url}
    
    def _extract_text(self, soup, selector):
        """Extract text from element"""
        try:
            element = soup.select_one(selector)
            return element.get_text().strip() if element else None
        except:
            return None
    
    def _extract_field(self, soup, field_name):
        """Extract field using dt:contains pattern"""
        try:
            dt_element = soup.find('dt', string=lambda text: text and field_name in text)
            if dt_element and dt_element.find_next_sibling('dd'):
                dd_element = dt_element.find_next_sibling('dd')
                if field_name == "Website":
                    link = dd_element.find('a')
                    return link.get('href') if link else dd_element.get_text().strip()
                else:
                    return dd_element.get_text().strip()
        except Exception as e:
            self.logger.warning(f"Failed to extract {field_name}: {e}")
        return None
    
    def _get_timestamp(self):
        """Get current timestamp"""
        return datetime.now().isoformat()


class LinkedInScraperService:
    """Service class for LinkedIn company scraping using Scrapfly API"""
    
    def __init__(self, scrapfly_api_key: str = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = scrapfly_api_key or os.getenv('SCRAPFLY_API_KEY')
        self.extractor = LinkedInDataExtractor()
        
        if SCRAPFLY_AVAILABLE and self.api_key:
            self.client = ScrapflyClient(key=self.api_key)
        else:
            self.client = None
            if not SCRAPFLY_AVAILABLE:
                self.logger.warning("Scrapfly SDK not available. Install with: pip install scrapfly-sdk")
            if not self.api_key:
                self.logger.warning("SCRAPFLY_API_KEY not set. Using mock data.")
        
    async def scrape_company(self, company_name: str) -> Dict[str, Any]:
        """
        Scrape LinkedIn company information using Scrapfly API
        
        Args:
            company_name: Name of the company to scrape
            
        Returns:
            Dict containing company information
        """
        # Construct LinkedIn company URL
        company_slug = company_name.lower().replace(" ", "-").replace("&", "and")
        company_url = f"https://www.linkedin.com/company/{company_slug}/"
        
        self.logger.info(f"Scraping LinkedIn for company: {company_name}")
        self.logger.info(f"Company URL: {company_url}")
        
        if not self.client:
            # Return mock data if Scrapfly not available
            return self._get_mock_data(company_name, company_url)
        
        try:
            # Use Scrapfly to scrape the LinkedIn page
            result = await self.client.scrape(
                ScrapeConfig(
                    url=company_url,
                    render_js=True,  # LinkedIn uses JavaScript
                    country="US",    # Use US proxy
                    wait_for_selector="h1",  # Wait for company name to load
                    timeout=30000,   # 30 second timeout
                )
            )
            
            if result.success:
                # Extract data from HTML
                data = self.extractor.extract_company_data(result.content, company_url)
                self.logger.info(f"Successfully scraped: {company_name}")
                return data
            else:
                self.logger.error(f"Scrapfly failed for {company_name}: {result.error}")
                return self._get_mock_data(company_name, company_url)
                
        except Exception as e:
            self.logger.error(f"Error scraping {company_name}: {e}")
            return self._get_mock_data(company_name, company_url)
    
    def _get_mock_data(self, company_name: str, company_url: str) -> Dict[str, Any]:
        """Return mock data when scraping fails"""
        company_slug = company_name.lower().replace(" ", "-").replace("&", "and")
        return {
            "company_name": company_name,
            "website": f"https://{company_slug}.com",
            "industry": "Technology",
            "size": "10,001+ employees",
            "hq_location": "Mountain View, CA",
            "company_type": "Public Company",
            "url": company_url,
            "scraped_at": self._get_timestamp(),
            "note": "Mock data - Scrapfly not configured"
        }
    
    def _get_timestamp(self):
        """Get current timestamp"""
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
