"""
Research service for automated data enrichment
"""

import structlog
from typing import Dict, Any, Optional, List
import asyncio
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import re
from urllib.parse import urljoin, urlparse

logger = structlog.get_logger(__name__)


class ResearchService:
    """Research service for automated data enrichment"""
    
    def __init__(self):
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    async def enrich_company_data(self, company_name: str, domain: str = None) -> Dict[str, Any]:
        """Enrich company data with external research"""
        try:
            self.logger.info("Starting company enrichment", company=company_name)
            
            # Initialize enrichment data
            enrichment_data = {
                "company_name": company_name,
                "domain": domain,
                "website": None,
                "description": None,
                "industry": None,
                "size": None,
                "location": None,
                "social_profiles": {},
                "news": [],
                "funding_info": None,
                "confidence": 0.0
            }
            
            # Search for company website
            if not domain:
                website = await self._find_company_website(company_name)
                enrichment_data["website"] = website
                domain = website
            
            # Get company information from website
            if domain:
                website_data = await self._scrape_company_website(domain)
                enrichment_data.update(website_data)
            
            # Search for social profiles
            social_profiles = await self._find_social_profiles(company_name)
            enrichment_data["social_profiles"] = social_profiles
            
            # Search for news and funding
            news = await self._search_company_news(company_name)
            enrichment_data["news"] = news
            
            # Calculate confidence score
            enrichment_data["confidence"] = self._calculate_confidence(enrichment_data)
            
            self.logger.info("Company enrichment completed", 
                           company=company_name, 
                           confidence=enrichment_data["confidence"])
            
            return {
                "success": True,
                "data": enrichment_data
            }
            
        except Exception as e:
            self.logger.error("Company enrichment failed", company=company_name, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
    
    async def _find_company_website(self, company_name: str) -> Optional[str]:
        """Find company website using search"""
        try:
            # Search for company website
            search_query = f"{company_name} official website"
            search_url = f"https://www.google.com/search?q={search_query}"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                await page.goto(search_url)
                await page.wait_for_load_state('networkidle')
                
                # Look for official website in search results
                links = await page.query_selector_all('a[href*="http"]')
                
                for link in links:
                    href = await link.get_attribute('href')
                    if href and not any(domain in href for domain in ['google.com', 'youtube.com', 'facebook.com', 'linkedin.com']):
                        # Check if it looks like an official website
                        if any(keyword in href.lower() for keyword in ['company', 'corp', 'inc', 'llc', 'ltd']):
                            await browser.close()
                            return href
                
                await browser.close()
                return None
                
        except Exception as e:
            self.logger.warning("Website search failed", company=company_name, error=str(e))
            return None
    
    async def _scrape_company_website(self, domain: str) -> Dict[str, Any]:
        """Scrape company information from website"""
        try:
            website_data = {
                "description": None,
                "industry": None,
                "size": None,
                "location": None
            }
            
            # Use Playwright for dynamic content
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                await page.goto(f"https://{domain}")
                await page.wait_for_load_state('networkidle')
                
                # Extract page content
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract description from meta tags
                description = soup.find('meta', attrs={'name': 'description'})
                if description:
                    website_data["description"] = description.get('content')
                
                # Look for industry keywords
                industry_keywords = [
                    'technology', 'software', 'healthcare', 'finance', 'retail',
                    'manufacturing', 'consulting', 'education', 'media', 'energy'
                ]
                
                page_text = soup.get_text().lower()
                for keyword in industry_keywords:
                    if keyword in page_text:
                        website_data["industry"] = keyword.title()
                        break
                
                # Look for company size indicators
                size_patterns = [
                    r'(\d+)\s*employees',
                    r'team\s*of\s*(\d+)',
                    r'(\d+)\s*people'
                ]
                
                for pattern in size_patterns:
                    match = re.search(pattern, page_text)
                    if match:
                        size = int(match.group(1))
                        if size < 50:
                            website_data["size"] = "Small (1-50)"
                        elif size < 200:
                            website_data["size"] = "Medium (51-200)"
                        else:
                            website_data["size"] = "Large (200+)"
                        break
                
                await browser.close()
                
            return website_data
            
        except Exception as e:
            self.logger.warning("Website scraping failed", domain=domain, error=str(e))
            return {}
    
    async def _find_social_profiles(self, company_name: str) -> Dict[str, str]:
        """Find company social media profiles"""
        try:
            social_profiles = {}
            
            # LinkedIn
            linkedin_url = await self._search_linkedin_company(company_name)
            if linkedin_url:
                social_profiles["linkedin"] = linkedin_url
            
            # Twitter
            twitter_url = await self._search_twitter_company(company_name)
            if twitter_url:
                social_profiles["twitter"] = twitter_url
            
            # Facebook
            facebook_url = await self._search_facebook_company(company_name)
            if facebook_url:
                social_profiles["facebook"] = facebook_url
            
            return social_profiles
            
        except Exception as e:
            self.logger.warning("Social profile search failed", company=company_name, error=str(e))
            return {}
    
    async def _search_linkedin_company(self, company_name: str) -> Optional[str]:
        """Search for LinkedIn company page"""
        try:
            search_url = f"https://www.linkedin.com/search/results/companies/?keywords={company_name}"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                await page.goto(search_url)
                await page.wait_for_load_state('networkidle')
                
                # Look for company page link
                company_link = await page.query_selector('a[href*="/company/"]')
                if company_link:
                    href = await company_link.get_attribute('href')
                    await browser.close()
                    return href
                
                await browser.close()
                return None
                
        except Exception as e:
            self.logger.warning("LinkedIn search failed", company=company_name, error=str(e))
            return None
    
    async def _search_twitter_company(self, company_name: str) -> Optional[str]:
        """Search for Twitter company profile"""
        try:
            search_url = f"https://twitter.com/search?q={company_name}&src=typed_query"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                await page.goto(search_url)
                await page.wait_for_load_state('networkidle')
                
                # Look for verified company account
                verified_account = await page.query_selector('[data-testid="UserAvatar-Container-verified"]')
                if verified_account:
                    profile_link = await verified_account.query_selector('a')
                    if profile_link:
                        href = await profile_link.get_attribute('href')
                        await browser.close()
                        return f"https://twitter.com{href}"
                
                await browser.close()
                return None
                
        except Exception as e:
            self.logger.warning("Twitter search failed", company=company_name, error=str(e))
            return None
    
    async def _search_facebook_company(self, company_name: str) -> Optional[str]:
        """Search for Facebook company page"""
        try:
            search_url = f"https://www.facebook.com/search/pages/?q={company_name}"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                await page.goto(search_url)
                await page.wait_for_load_state('networkidle')
                
                # Look for official page
                official_page = await page.query_selector('[data-testid="official-page"]')
                if official_page:
                    page_link = await official_page.query_selector('a')
                    if page_link:
                        href = await page_link.get_attribute('href')
                        await browser.close()
                        return href
                
                await browser.close()
                return None
                
        except Exception as e:
            self.logger.warning("Facebook search failed", company=company_name, error=str(e))
            return None
    
    async def _search_company_news(self, company_name: str) -> List[Dict[str, Any]]:
        """Search for recent company news"""
        try:
            news = []
            
            # Search for news using Google News
            search_url = f"https://news.google.com/search?q={company_name}&hl=en&gl=US&ceid=US:en"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                await page.goto(search_url)
                await page.wait_for_load_state('networkidle')
                
                # Extract news articles
                articles = await page.query_selector_all('article')
                
                for article in articles[:5]:  # Limit to 5 articles
                    try:
                        title_element = await article.query_selector('h3')
                        link_element = await article.query_selector('a')
                        
                        if title_element and link_element:
                            title = await title_element.inner_text()
                            link = await link_element.get_attribute('href')
                            
                            news.append({
                                "title": title,
                                "url": link,
                                "source": "Google News"
                            })
                    except Exception as e:
                        continue
                
                await browser.close()
            
            return news
            
        except Exception as e:
            self.logger.warning("News search failed", company=company_name, error=str(e))
            return []
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for enrichment data"""
        score = 0.0
        
        # Website found
        if data.get("website"):
            score += 0.3
        
        # Description found
        if data.get("description"):
            score += 0.2
        
        # Industry identified
        if data.get("industry"):
            score += 0.2
        
        # Social profiles found
        social_count = len(data.get("social_profiles", {}))
        score += min(social_count * 0.1, 0.2)
        
        # News found
        if data.get("news"):
            score += 0.1
        
        return min(score, 1.0)
