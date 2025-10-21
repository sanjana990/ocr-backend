#!/usr/bin/env python3
"""
Database service for saving LinkedIn scraped data to Supabase
"""

import os
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseService:
    """Service for database operations"""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            print("⚠️ Supabase credentials not found. Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.")
            self.available = False
        else:
            self.available = True
            print("✅ Database service initialized with Supabase")
    
    async def save_business_card_data(self, card_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save business card OCR data to Supabase
        
        Args:
            card_data: Dictionary containing business card information
            
        Returns:
            Saved data or None if failed
        """
        if not self.available:
            print("❌ Database service not available")
            return None
        
        try:
            # Prepare data for Supabase
            supabase_data = {
                "name": card_data.get("name", ""),
                "title": card_data.get("title"),
                "company": card_data.get("company"),
                "email": card_data.get("email"),
                "phone": card_data.get("phone"),
                "address": card_data.get("address"),
                "website": card_data.get("website"),
                "raw_text": card_data.get("raw_text"),
                "confidence": card_data.get("confidence", 0.0),
                "engine_used": card_data.get("engine_used"),
                "qr_codes": card_data.get("qr_codes", []),
                "qr_count": card_data.get("qr_count", 0),
                "extracted_at": card_data.get("extracted_at", datetime.now().isoformat())
            }
            
            # Make API call to Supabase
            async with aiohttp.ClientSession() as session:
                url = f"{self.supabase_url}/rest/v1/business_cards"
                headers = {
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
                
                async with session.post(url, json=supabase_data, headers=headers) as response:
                    if response.status == 201:
                        result = await response.json()
                        print(f"✅ Business card data saved to database: {card_data.get('name')}")
                        return result[0] if result else None
                    else:
                        error_text = await response.text()
                        print(f"❌ Failed to save business card: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            print(f"❌ Database save error: {e}")
            return None

    async def save_linkedin_company(self, company_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save LinkedIn company data to Supabase
        
        Args:
            company_data: Dictionary containing company information
            
        Returns:
            Saved data or None if failed
        """
        if not self.available:
            print("❌ Database service not available")
            return None
        
        try:
            # Prepare data for Supabase
            supabase_data = {
                "company_name": company_data.get("company_name", ""),
                "website": company_data.get("website"),
                "industry": company_data.get("industry"),
                "company_size": company_data.get("size"),
                "hq_location": company_data.get("hq_location"),
                "company_type": company_data.get("company_type"),
                "linkedin_url": company_data.get("url"),
                "scraped_at": company_data.get("scraped_at", datetime.now().isoformat())
            }
            
            # Make API call to Supabase
            async with aiohttp.ClientSession() as session:
                url = f"{self.supabase_url}/rest/v1/linkedin_companies"
                headers = {
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
                
                async with session.post(url, json=supabase_data, headers=headers) as response:
                    if response.status == 201:
                        result = await response.json()
                        print(f"✅ LinkedIn company saved to database: {company_data.get('company_name')}")
                        return result[0] if result else None
                    else:
                        error_text = await response.text()
                        print(f"❌ Failed to save to database: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            print(f"❌ Database save error: {e}")
            return None
    
    async def get_linkedin_companies(self) -> Optional[list]:
        """Get all LinkedIn companies from database"""
        if not self.available:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.supabase_url}/rest/v1/linkedin_companies"
                headers = {
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"❌ Failed to fetch companies: {response.status}")
                        return None
                        
        except Exception as e:
            print(f"❌ Database fetch error: {e}")
            return None

    async def save_crawl_data(self, crawl_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save crawl data to Supabase
        
        Args:
            crawl_data: Dictionary containing crawl information
            
        Returns:
            Saved data or None if failed
        """
        if not self.available:
            print("❌ Database service not available")
            return None
        
        try:
            # Prepare data for Supabase
            supabase_data = {
                "company_name": crawl_data.get("company_name", ""),
                "url": crawl_data.get("url", ""),
                "platform": crawl_data.get("platform", ""),
                "content": crawl_data.get("content", ""),
                "ai_extracted_data": json.dumps(crawl_data.get("ai_extracted_data", {})),
                "crawl_time": crawl_data.get("crawl_time", 0),
                "crawled_at": crawl_data.get("crawled_at", datetime.now().isoformat()),
                "created_at": datetime.now().isoformat()
            }
            
            # Insert into Supabase
            url = f"{self.supabase_url}/rest/v1/crawl_data"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=supabase_data) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        print(f"✅ Crawl data saved to Supabase: {crawl_data.get('company_name')}")
                        return result[0] if result else None
                    else:
                        error_text = await response.text()
                        print(f"❌ Failed to save crawl data: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            print(f"❌ Database save error: {e}")
            return None

# Global instance
database_service = DatabaseService()
