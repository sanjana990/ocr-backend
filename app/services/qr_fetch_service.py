"""
QR Code URL Fetching Service
Fetches additional details from QR code URLs for business card enhancement
"""

import asyncio
import aiohttp
import structlog
from typing import Dict, Any, Optional, List
import re
from urllib.parse import urlparse
import json

logger = structlog.get_logger(__name__)


class QRFetchService:
    """Service to fetch additional details from QR code URLs"""
    
    def __init__(self):
        self.logger = logger
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=10, connect=5)
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_url_details(self, url: str) -> Dict[str, Any]:
        """Fetch additional details from a URL"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        
        try:
            self.logger.info("🌐 Fetching URL details", url=url)
            
            # Add headers to appear as a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    return await self._parse_webpage_content(content, url)
                else:
                    self.logger.warning("Failed to fetch URL", url=url, status=response.status)
                    return {"error": f"HTTP {response.status}", "url": url}
                    
        except asyncio.TimeoutError:
            self.logger.warning("Timeout fetching URL", url=url)
            return {"error": "timeout", "url": url}
        except Exception as e:
            self.logger.warning("Error fetching URL", url=url, error=str(e))
            return {"error": str(e), "url": url}
    
    async def _parse_webpage_content(self, content: str, url: str) -> Dict[str, Any]:
        """Parse webpage content for business card relevant information"""
        try:
            # Extract title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""
            
            # Extract meta description
            desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', content, re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else ""
            
            # Extract business information
            business_info = self._extract_business_info(content)
            
            # Extract contact information
            contact_info = self._extract_contact_info(content)
            
            # Extract social media links
            social_links = self._extract_social_links(content)
            
            return {
                "url": url,
                "title": title,
                "description": description,
                "business_info": business_info,
                "contact_info": contact_info,
                "social_links": social_links,
                "success": True
            }
            
        except Exception as e:
            self.logger.warning("Error parsing webpage content", url=url, error=str(e))
            return {"error": str(e), "url": url}
    
    def _extract_business_info(self, content: str) -> Dict[str, Any]:
        """Extract business-related information from webpage content"""
        business_info = {}
        
        # Look for company name patterns
        company_patterns = [
            r'<h1[^>]*>(.*?)</h1>',
            r'<h2[^>]*>(.*?)</h2>',
            r'class=["\']company["\'][^>]*>(.*?)<',
            r'class=["\']business["\'][^>]*>(.*?)<',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
                if len(text) > 3 and len(text) < 100:  # Reasonable company name length
                    business_info['company_name'] = text
                    break
        
        # Look for business type indicators
        business_types = ['company', 'corp', 'inc', 'llc', 'ltd', 'business', 'enterprise']
        for btype in business_types:
            if btype in content.lower():
                business_info['business_type'] = btype
                break
        
        return business_info
    
    def _extract_contact_info(self, content: str) -> Dict[str, Any]:
        """Extract contact information from webpage content"""
        contact_info = {}
        
        # Email patterns
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, content)
        if emails:
            contact_info['emails'] = list(set(emails))  # Remove duplicates
        
        # Phone patterns
        phone_patterns = [
            r'\+?[\d\s\-\(\)]{10,}',
            r'\(\d{3}\)\s*\d{3}-\d{4}',
            r'\d{3}-\d{3}-\d{4}',
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, content))
        
        if phones:
            contact_info['phones'] = list(set(phones))  # Remove duplicates
        
        # Address patterns
        address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)'
        addresses = re.findall(address_pattern, content, re.IGNORECASE)
        if addresses:
            contact_info['addresses'] = list(set(addresses))
        
        return contact_info
    
    def _extract_social_links(self, content: str) -> Dict[str, List[str]]:
        """Extract social media links from webpage content"""
        social_links = {}
        
        # Social media patterns
        social_patterns = {
            'linkedin': r'https?://(?:www\.)?linkedin\.com/[^\s"\'<>]+',
            'twitter': r'https?://(?:www\.)?twitter\.com/[^\s"\'<>]+',
            'facebook': r'https?://(?:www\.)?facebook\.com/[^\s"\'<>]+',
            'instagram': r'https?://(?:www\.)?instagram\.com/[^\s"\'<>]+',
            'youtube': r'https?://(?:www\.)?youtube\.com/[^\s"\'<>]+',
        }
        
        for platform, pattern in social_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                social_links[platform] = list(set(matches))  # Remove duplicates
        
        return social_links
    
    async def fetch_multiple_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Fetch details from multiple URLs concurrently"""
        if not urls:
            return {}
        
        self.logger.info(f"🌐 Fetching details from {len(urls)} URLs")
        
        # Create tasks for concurrent fetching
        tasks = [self.fetch_url_details(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                combined_results[urls[i]] = {"error": str(result), "url": urls[i]}
            else:
                combined_results[urls[i]] = result
        
        return combined_results


# Standalone function for easy integration
async def fetch_qr_url_details(url: str) -> Dict[str, Any]:
    """Standalone function to fetch URL details from a QR code"""
    async with QRFetchService() as fetch_service:
        return await fetch_service.fetch_url_details(url)


async def fetch_multiple_qr_urls(urls: List[str]) -> Dict[str, Any]:
    """Standalone function to fetch details from multiple QR code URLs"""
    async with QRFetchService() as fetch_service:
        return await fetch_service.fetch_multiple_urls(urls)
