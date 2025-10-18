"""
Advanced URL content extraction service using AI
"""
import requests
import logging
from typing import Dict, Any, Optional
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
import re

logger = logging.getLogger(__name__)

class URLContentService:
    def __init__(self):
        """Initialize the URL content service"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; BusinessCardScanner/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    async def extract_contact_info(self, url: str) -> Dict[str, Any]:
        """
        Extract contact information from a URL using AI and web scraping
        """
        try:
            if not BeautifulSoup:
                return {
                    'success': False,
                    'url': url,
                    'error': 'BeautifulSoup not available'
                }
            
            # Fetch the webpage
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic info
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ''
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '').strip() if meta_desc else ''
            
            # Get all text content
            text_content = soup.get_text()
            
            # Extract structured information
            contact_info = self._extract_contact_details(text_content)
            
            # Extract social media links
            social_links = self._extract_social_links(soup)
            
            # Extract business information
            business_info = self._extract_business_info(text_content)
            
            return {
                'success': True,
                'url': url,
                'title': title_text,
                'description': description,
                'contact_info': contact_info,
                'social_links': social_links,
                'business_info': business_info,
                'raw_content': text_content[:1000] + '...' if len(text_content) > 1000 else text_content
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {str(e)}")
            return {
                'success': False,
                'url': url,
                'error': f"Failed to fetch URL: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return {
                'success': False,
                'url': url,
                'error': f"Error processing content: {str(e)}"
            }
    
    def _extract_contact_details(self, text: str) -> Dict[str, Any]:
        """Extract contact details from text content"""
        contact_info = {
            'emails': [],
            'phones': [],
            'names': [],
            'addresses': []
        }
        
        # Extract emails
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        contact_info['emails'] = list(set(emails))
        
        # Extract phone numbers
        phone_patterns = [
            r'\+?[\d\s\-\(\)]{10,}',  # General phone pattern
            r'\(\d{3}\)\s*\d{3}-\d{4}',  # (123) 456-7890
            r'\d{3}-\d{3}-\d{4}',  # 123-456-7890
            r'\+\d{1,3}\s?\d{1,4}\s?\d{1,4}\s?\d{1,9}'  # International format
        ]
        
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        contact_info['phones'] = list(set([p.strip() for p in phones if len(p.strip()) >= 10]))
        
        # Extract potential names (improved pattern)
        name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # First M. Last
            r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b'  # First Middle Last
        ]
        
        names = []
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        # Filter out common false positives
        filtered_names = []
        for name in names:
            if not any(word.lower() in ['company', 'inc', 'llc', 'corp', 'ltd', 'email', 'phone', 'address'] 
                      for word in name.split()):
                filtered_names.append(name)
        
        contact_info['names'] = list(set(filtered_names))
        
        # Extract addresses
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)',
            r'\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}'
        ]
        
        addresses = []
        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            addresses.extend(matches)
        
        contact_info['addresses'] = list(set(addresses))
        
        return contact_info
    
    def _extract_social_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract social media links from HTML"""
        social_links = {}
        
        # Find all links
        links = soup.find_all('a', href=True)
        
        social_patterns = {
            'linkedin': r'linkedin\.com/in/',
            'twitter': r'twitter\.com/',
            'facebook': r'facebook\.com/',
            'instagram': r'instagram\.com/',
            'youtube': r'youtube\.com/',
            'github': r'github\.com/'
        }
        
        for link in links:
            href = link['href']
            for platform, pattern in social_patterns.items():
                if re.search(pattern, href, re.IGNORECASE):
                    social_links[platform] = href
                    break
        
        return social_links
    
    def _extract_business_info(self, text: str) -> Dict[str, Any]:
        """Extract business information from text"""
        business_info = {
            'company_name': '',
            'industry': '',
            'services': [],
            'description': ''
        }
        
        # Extract company names (look for common business suffixes)
        company_pattern = r'([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Ltd|Company|Co\.|Pvt|Private|Group|Solutions|Technologies|Systems))'
        companies = re.findall(company_pattern, text, re.IGNORECASE)
        if companies:
            business_info['company_name'] = companies[0].strip()
        
        # Extract industry keywords
        industry_keywords = [
            'technology', 'software', 'consulting', 'marketing', 'finance',
            'healthcare', 'education', 'retail', 'manufacturing', 'services'
        ]
        
        found_industries = []
        for keyword in industry_keywords:
            if keyword.lower() in text.lower():
                found_industries.append(keyword)
        
        business_info['industry'] = ', '.join(found_industries)
        
        return business_info

# Create singleton instance
url_content_service = URLContentService()
