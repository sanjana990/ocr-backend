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

# Global instance
database_service = DatabaseService()
