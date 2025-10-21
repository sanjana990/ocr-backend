#!/usr/bin/env python3
"""
MongoDB service for storing enriched company data
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️ MongoDB not available. Install with: pip install motor")

# Load environment variables
load_dotenv()

class MongoDBService:
    """Service for MongoDB operations"""
    
    def __init__(self):
        self.mongodb_url = os.getenv('MONGODB_URL')
        self.database_name = os.getenv('MONGODB_DATABASE', 'ocr_platform')
        
        if not self.mongodb_url or not MONGODB_AVAILABLE:
            print("⚠️ MongoDB not configured. Set MONGODB_URL environment variable.")
            self.available = False
        else:
            try:
                self.client = AsyncIOMotorClient(self.mongodb_url)
                self.db = self.client[self.database_name]
                self.available = True
                print("✅ MongoDB service initialized")
            except Exception as e:
                print(f"❌ MongoDB connection failed: {e}")
                self.available = False
    
    async def save_enriched_company_data(self, company_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save enriched company data to MongoDB
        
        Args:
            company_data: Dictionary containing enriched company information
            
        Returns:
            Saved data or None if failed
        """
        if not self.available:
            print("❌ MongoDB service not available")
            return None
        
        try:
            # Prepare data for MongoDB
            mongo_data = {
                "company_name": company_data.get("company_name", ""),
                "website": company_data.get("website"),
                "industry": company_data.get("industry"),
                "size": company_data.get("size"),
                "hq_location": company_data.get("hq_location"),
                "company_type": company_data.get("company_type"),
                "linkedin_url": company_data.get("linkedin_url"),
                "twitter_url": company_data.get("twitter_url"),
                "facebook_url": company_data.get("facebook_url"),
                "description": company_data.get("description"),
                "founded_year": company_data.get("founded_year"),
                "revenue": company_data.get("revenue"),
                "technologies": company_data.get("technologies", []),
                "contacts": company_data.get("contacts", []),
                "web_scraped_data": company_data.get("web_scraped_data", {}),
                "source": company_data.get("source", "unknown"),
                "enriched_at": company_data.get("enriched_at", datetime.now().isoformat()),
                "created_at": datetime.now().isoformat()
            }
            
            # Insert into MongoDB
            result = await self.db.enriched_companies.insert_one(mongo_data)
            
            if result.inserted_id:
                print(f"✅ Enriched company data saved to MongoDB: {company_data.get('company_name')}")
                return {"id": str(result.inserted_id), **mongo_data}
            else:
                print("❌ Failed to save enriched company data")
                return None
                
        except Exception as e:
            print(f"❌ MongoDB save error: {e}")
            return None
    
    async def get_enriched_company(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Get enriched company data from MongoDB"""
        if not self.available:
            return None
        
        try:
            company = await self.db.enriched_companies.find_one(
                {"company_name": {"$regex": company_name, "$options": "i"}}
            )
            return company
        except Exception as e:
            print(f"❌ MongoDB fetch error: {e}")
            return None
    
    async def get_all_enriched_companies(self) -> List[Dict[str, Any]]:
        """Get all enriched companies from MongoDB"""
        if not self.available:
            return []
        
        try:
            companies = []
            async for company in self.db.enriched_companies.find():
                companies.append(company)
            return companies
        except Exception as e:
            print(f"❌ MongoDB fetch error: {e}")
            return []
    
    async def save_web_scraped_data(self, company_name: str, web_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Save web scraped data for a company"""
        if not self.available:
            return None
        
        try:
            web_scraped_data = {
                "company_name": company_name,
                "website_data": web_data,
                "scraped_at": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat()
            }
            
            result = await self.db.web_scraped_data.insert_one(web_scraped_data)
            
            if result.inserted_id:
                print(f"✅ Web scraped data saved for: {company_name}")
                return {"id": str(result.inserted_id), **web_scraped_data}
            else:
                print("❌ Failed to save web scraped data")
                return None
                
        except Exception as e:
            print(f"❌ MongoDB save error: {e}")
            return None

    async def save_crawl_data(self, crawl_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save crawl data to MongoDB
        
        Args:
            crawl_data: Dictionary containing crawl information
            
        Returns:
            Saved data or None if failed
        """
        if not self.available:
            print("❌ MongoDB service not available")
            return None
        
        try:
            # Prepare data for MongoDB
            mongo_data = {
                "company_name": crawl_data.get("company_name", ""),
                "url": crawl_data.get("url", ""),
                "platform": crawl_data.get("platform", ""),
                "content": crawl_data.get("content", ""),
                "ai_extracted_data": crawl_data.get("ai_extracted_data", {}),
                "crawl_time": crawl_data.get("crawl_time", 0),
                "crawled_at": crawl_data.get("crawled_at", datetime.now().isoformat()),
                "created_at": datetime.now().isoformat()
            }
            
            # Insert into MongoDB
            result = await self.db.crawl_data.insert_one(mongo_data)
            
            if result.inserted_id:
                print(f"✅ Crawl data saved to MongoDB: {crawl_data.get('company_name')}")
                return {"id": str(result.inserted_id), **mongo_data}
            else:
                print("❌ Failed to save crawl data to MongoDB")
                return None
                
        except Exception as e:
            print(f"❌ MongoDB save error: {e}")
            return None

# Global instance
mongodb_service = MongoDBService()
