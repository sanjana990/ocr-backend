#!/usr/bin/env python3
"""
Apollo.io Company Data Service - Replace LinkedIn scraping with Apollo.io API
"""

import asyncio
import json
import logging
import os
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp

class ApolloService:
    """Service class for Apollo.io company data retrieval"""
    
    def __init__(self, api_key: str = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv('APOLLO_API_KEY')
        self.base_url = "https://api.apollo.io/v1"
        
        if not self.api_key:
            self.logger.warning("Apollo.io API key not found. Set APOLLO_API_KEY environment variable.")
            self.available = False
        else:
            self.available = True
    
    async def search_company(self, company_name: str) -> Dict[str, Any]:
        """
        Search for company information using Apollo.io API
        
        Args:
            company_name: Name of the company to search
            
        Returns:
            Dict containing company information
        """
        if not self.available:
            return self._get_mock_data(company_name)
        
        try:
            self.logger.info(f"Searching Apollo.io for company: {company_name}")
            
            # Search for company using Apollo.io API (free tier endpoint)
            search_url = f"{self.base_url}/organizations/search"
            
            headers = {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'X-Api-Key': self.api_key
            }
            
            payload = {
                'q_organization_domains': company_name,
                'page': 1,
                'per_page': 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(search_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        companies = data.get('organizations', [])
                        
                        if companies:
                            company = companies[0]
                            self.logger.info(f"✅ Apollo.io API success: Found {company.get('name', 'Unknown')}")
                            return self._format_company_data(company, company_name)
                        else:
                            self.logger.warning(f"No company found for: {company_name}")
                            return self._get_mock_data(company_name)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Apollo.io API error: {response.status} - {error_text}")
                        return self._get_mock_data(company_name)
                        
        except Exception as e:
            self.logger.error(f"Error searching Apollo.io for {company_name}: {e}")
            return self._get_mock_data(company_name)
    
    def _format_company_data(self, company: Dict[str, Any], original_name: str) -> Dict[str, Any]:
        """Format Apollo.io company data into our standard format"""
        return {
            "company_name": company.get('name', original_name),
            "website": company.get('website_url'),
            "industry": company.get('industry'),
            "size": self._format_company_size(company.get('estimated_num_employees')),
            "hq_location": self._format_location(company.get('city'), company.get('state')),
            "company_type": company.get('organization_type'),
            "linkedin_url": company.get('linkedin_url'),
            "twitter_url": company.get('twitter_url'),
            "facebook_url": company.get('facebook_url'),
            "description": company.get('short_description'),
            "founded_year": company.get('founded_year'),
            "revenue": company.get('annual_revenue'),
            "technologies": company.get('technologies', []),
            "url": company.get('website_url'),
            "scraped_at": self._get_timestamp(),
            "source": "apollo.io"
        }
    
    def _format_company_size(self, employee_count: Optional[int]) -> str:
        """Format employee count into readable size"""
        if not employee_count:
            return "Unknown"
        
        if employee_count < 10:
            return "1-10 employees"
        elif employee_count < 50:
            return "11-50 employees"
        elif employee_count < 200:
            return "51-200 employees"
        elif employee_count < 500:
            return "201-500 employees"
        elif employee_count < 1000:
            return "501-1000 employees"
        elif employee_count < 5000:
            return "1001-5000 employees"
        elif employee_count < 10000:
            return "5001-10000 employees"
        else:
            return "10000+ employees"
    
    def _format_location(self, city: Optional[str], state: Optional[str]) -> str:
        """Format location from city and state"""
        location_parts = []
        if city:
            location_parts.append(city)
        if state:
            location_parts.append(state)
        return ", ".join(location_parts) if location_parts else "Unknown"
    
    def _get_mock_data(self, company_name: str) -> Dict[str, Any]:
        """Return mock data when Apollo.io is not available"""
        company_slug = company_name.lower().replace(" ", "-").replace("&", "and")
        return {
            "company_name": company_name,
            "website": f"https://{company_slug}.com",
            "industry": "Technology",
            "size": "10,001+ employees",
            "hq_location": "Mountain View, CA",
            "company_type": "Public Company",
            "linkedin_url": f"https://www.linkedin.com/company/{company_slug}/",
            "twitter_url": None,
            "facebook_url": None,
            "description": f"Leading technology company specializing in {company_name.lower()} solutions",
            "founded_year": 2010,
            "revenue": "$10M - $50M",
            "technologies": ["Python", "JavaScript", "Cloud Computing"],
            "url": f"https://{company_slug}.com",
            "scraped_at": self._get_timestamp(),
            "source": "apollo.io (mock data)",
            "note": "Mock data - Apollo.io not configured"
        }
    
    def _get_timestamp(self):
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    async def search_multiple_companies(self, company_names: List[str]) -> List[Dict[str, Any]]:
        """Search for multiple companies"""
        results = []
        
        for company_name in company_names:
            try:
                result = await self.search_company(company_name)
                results.append(result)
                
                # Add small delay to respect rate limits
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error searching for {company_name}: {e}")
                results.append(self._get_mock_data(company_name))
        
        return results
    
    async def get_company_contacts(self, company_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get contacts for a company"""
        if not self.available:
            return []
        
        try:
            # Search for people at the company (not available on free tier)
            search_url = f"{self.base_url}/people/search"
            
            headers = {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'X-Api-Key': self.api_key
            }
            
            payload = {
                'q_organization_domains': company_name,
                'page': 1,
                'per_page': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(search_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        people = data.get('people', [])
                        
                        contacts = []
                        for person in people:
                            contact = {
                                "name": person.get('name'),
                                "title": person.get('title'),
                                "email": person.get('email'),
                                "phone": person.get('phone_numbers', [{}])[0].get('sanitized_number') if person.get('phone_numbers') else None,
                                "linkedin_url": person.get('linkedin_url'),
                                "twitter_url": person.get('twitter_url'),
                                "company": person.get('organization', {}).get('name'),
                                "location": person.get('city', '') + ', ' + person.get('state', '') if person.get('city') else None
                            }
                            contacts.append(contact)
                        
                        return contacts
                    else:
                        self.logger.error(f"Apollo.io contacts API error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error getting contacts for {company_name}: {e}")
            return []


# Global instance
apollo_service = ApolloService()
