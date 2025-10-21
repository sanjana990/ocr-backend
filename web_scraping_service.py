#!/usr/bin/env python3
"""
Web scraping service using Playwright for company website analysis
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not available. Install with: pip install playwright")

class WebScrapingService:
    """Service for web scraping company websites using Playwright"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available = PLAYWRIGHT_AVAILABLE
    
    async def scrape_company_website(self, company_name: str, website_url: str = None) -> Dict[str, Any]:
        """
        Scrape company website for additional information
        
        Args:
            company_name: Name of the company
            website_url: Company website URL (optional)
            
        Returns:
            Dictionary containing scraped website data
        """
        if not self.available:
            return self._get_mock_web_data(company_name)
        
        try:
            # If no website URL provided, try to find it
            if not website_url:
                website_url = await self._find_company_website(company_name)
            
            if not website_url:
                return self._get_mock_web_data(company_name)
            
            self.logger.info(f"Scraping website for {company_name}: {website_url}")
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set user agent
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                
                try:
                    await page.goto(website_url, wait_until='networkidle', timeout=30000)
                    
                    # Wait for content to load
                    await page.wait_for_timeout(2000)
                    
                    # Extract page data
                    page_data = await self._extract_page_data(page, company_name)
                    
                    await browser.close()
                    
                    return {
                        "company_name": company_name,
                        "website_url": website_url,
                        "scraped_at": datetime.now().isoformat(),
                        "source": "web_scraping",
                        **page_data
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error scraping {website_url}: {e}")
                    await browser.close()
                    return self._get_mock_web_data(company_name)
                    
        except Exception as e:
            self.logger.error(f"Web scraping failed for {company_name}: {e}")
            return self._get_mock_web_data(company_name)
    
    async def _extract_page_data(self, page, company_name: str) -> Dict[str, Any]:
        """Extract data from the webpage"""
        try:
            # Get page content
            title = await page.title()
            content = await page.content()
            
            # Extract meta description
            meta_description = await page.evaluate("""
                () => {
                    const meta = document.querySelector('meta[name="description"]');
                    return meta ? meta.getAttribute('content') : null;
                }
            """)
            
            # Extract social media links
            social_links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    const social = {};
                    
                    links.forEach(link => {
                        const href = link.href.toLowerCase();
                        if (href.includes('linkedin.com')) social.linkedin = link.href;
                        if (href.includes('twitter.com') || href.includes('x.com')) social.twitter = link.href;
                        if (href.includes('facebook.com')) social.facebook = link.href;
                        if (href.includes('instagram.com')) social.instagram = link.href;
                    });
                    
                    return social;
                }
            """)
            
            # Extract contact information
            contact_info = await page.evaluate("""
                () => {
                    const text = document.body.innerText;
                    const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
                    const phoneRegex = /(\\+?1[-.\\s]?)?\\(?([0-9]{3})\\)?[-.\\s]?([0-9]{3})[-.\\s]?([0-9]{4})/g;
                    
                    return {
                        emails: text.match(emailRegex) || [],
                        phones: text.match(phoneRegex) || []
                    };
                }
            """)
            
            # Extract company information
            company_info = await self._extract_company_info(content, company_name)
            
            return {
                "title": title,
                "meta_description": meta_description,
                "social_links": social_links,
                "contact_info": contact_info,
                "company_info": company_info,
                "page_size": len(content),
                "scraped_successfully": True
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting page data: {e}")
            return {
                "title": f"{company_name} Website",
                "scraped_successfully": False,
                "error": str(e)
            }
    
    def _extract_company_info(self, content: str, company_name: str) -> Dict[str, Any]:
        """Extract company information from page content"""
        content_lower = content.lower()
        
        # Industry keywords
        industry_keywords = [
            'technology', 'software', 'healthcare', 'finance', 'retail',
            'manufacturing', 'consulting', 'education', 'media', 'energy',
            'automotive', 'aerospace', 'pharmaceutical', 'real estate'
        ]
        
        detected_industry = None
        for keyword in industry_keywords:
            if keyword in content_lower:
                detected_industry = keyword.title()
                break
        
        # Company size indicators
        size_indicators = [
            r'(\d+)\s*employees',
            r'team\s*of\s*(\d+)',
            r'(\d+)\s*people',
            r'founded\s*in\s*(\d{4})'
        ]
        
        detected_size = None
        founded_year = None
        
        for pattern in size_indicators:
            matches = re.findall(pattern, content_lower)
            if matches:
                if 'employees' in pattern or 'people' in pattern:
                    detected_size = f"{matches[0]} employees"
                elif 'founded' in pattern:
                    founded_year = matches[0]
        
        return {
            "industry": detected_industry,
            "size": detected_size,
            "founded_year": founded_year,
            "content_analysis": {
                "has_about_page": 'about' in content_lower,
                "has_contact_page": 'contact' in content_lower,
                "has_careers_page": 'careers' in content_lower or 'jobs' in content_lower,
                "has_products_page": 'products' in content_lower or 'services' in content_lower
            }
        }
    
    async def _find_company_website(self, company_name: str) -> Optional[str]:
        """Try to find company website using search"""
        try:
            # Simple approach: construct likely website
            company_slug = company_name.lower().replace(" ", "").replace("&", "and")
            potential_urls = [
                f"https://www.{company_slug}.com",
                f"https://{company_slug}.com",
                f"https://www.{company_slug}.org",
                f"https://{company_slug}.org"
            ]
            
            # Test each URL
            for url in potential_urls:
                try:
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(headless=True)
                        page = await browser.new_page()
                        
                        try:
                            await page.goto(url, wait_until='networkidle', timeout=10000)
                            await browser.close()
                            return url
                        except:
                            await browser.close()
                            continue
                except:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding website for {company_name}: {e}")
            return None
    
    def _get_mock_web_data(self, company_name: str) -> Dict[str, Any]:
        """Return mock web data when scraping fails"""
        return {
            "company_name": company_name,
            "website_url": f"https://{company_name.lower().replace(' ', '')}.com",
            "title": f"{company_name} - Official Website",
            "meta_description": f"Official website of {company_name}",
            "social_links": {
                "linkedin": f"https://linkedin.com/company/{company_name.lower().replace(' ', '')}",
                "twitter": f"https://twitter.com/{company_name.lower().replace(' ', '')}"
            },
            "contact_info": {
                "emails": [f"info@{company_name.lower().replace(' ', '')}.com"],
                "phones": []
            },
            "company_info": {
                "industry": "Technology",
                "size": "100+ employees",
                "founded_year": "2010"
            },
            "scraped_at": datetime.now().isoformat(),
            "source": "web_scraping (mock data)",
            "note": "Mock data - Web scraping not available"
        }

# Global instance
web_scraping_service = WebScrapingService()
