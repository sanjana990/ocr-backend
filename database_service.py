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
    
    async def save_business_card_data(self, card_data: Dict[str, Any], image_data: Optional[bytes] = None, image_filename: Optional[str] = None, image_content_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Save business card OCR data to Supabase with image stored in Supabase Storage
        
        Args:
            card_data: Dictionary containing business card information
            image_data: Optional binary image data
            image_filename: Optional original filename
            image_content_type: Optional MIME type
            
        Returns:
            Saved data or None if failed
        """
        if not self.available:
            print("❌ Database service not available")
            return None
        
        try:
            image_url = None
            
            # Upload image to Supabase Storage if provided
            if image_data and image_filename:
                try:
                    # Generate unique filename
                    import uuid
                    import time
                    file_extension = image_filename.split('.')[-1] if '.' in image_filename else 'jpg'
                    unique_filename = f"{uuid.uuid4()}.{file_extension}"
                    
                    # Upload to Supabase Storage
                    storage_url = f"{self.supabase_url}/storage/v1/object/business-card-images/scanned-cards/{unique_filename}"
                    
                    headers = {
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": image_content_type or "image/jpeg"
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(storage_url, data=image_data, headers=headers) as response:
                            if response.status == 200:
                                # Get public URL
                                image_url = f"{self.supabase_url}/storage/v1/object/public/business-card-images/scanned-cards/{unique_filename}"
                                print(f"✅ Image uploaded to Supabase Storage: {unique_filename}")
                            else:
                                error_text = await response.text()
                                print(f"❌ Failed to upload image: {response.status} - {error_text}")
                                
                except Exception as e:
                    print(f"❌ Image upload error: {e}")
            
            # Prepare data for Supabase
            supabase_data = {
                "name": card_data.get("name", ""),
                "title": card_data.get("title"),
                "phone": card_data.get("phone"),
                "company": card_data.get("company"),
                "email": card_data.get("email"),
                "website": card_data.get("website"),
                "address": card_data.get("address"),
                "social_media": json.dumps(card_data.get("social_media", {})),
                "qr_codes": json.dumps(card_data.get("qr_codes", [])),
                "additional_info": json.dumps(card_data.get("additional_info", {})),
                "image": image_url  # Store URL instead of base64
            }
            
            # Make API call to Supabase
            async with aiohttp.ClientSession() as session:
                url = f"{self.supabase_url}/rest/v1/scanned_cards"
                headers = {
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation"
                }
                
                async with session.post(url, json=supabase_data, headers=headers) as response:
                    if response.status == 201:
                        result = await response.json()
                        print(f"✅ Scanned card data saved to database: {card_data.get('name')}")
                        return result[0] if result else None
                    else:
                        error_text = await response.text()
                        print(f"❌ Failed to save scanned card: {response.status} - {error_text}")
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

    async def save_company_data(self, company_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save enriched company data to Supabase
        
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
                "description": company_data.get("description"),
                "products": company_data.get("products"),
                "location": company_data.get("location"),
                "industry": company_data.get("industry"),
                "num_of_emp": company_data.get("num_of_emp"),
                "revenue": company_data.get("revenue"),
                "market_share": company_data.get("market_share")
            }
            
            # Insert into Supabase
            url = f"{self.supabase_url}/rest/v1/company_enrichment"
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
                        print(f"✅ Company data saved to Supabase: {company_data.get('company_name')}")
                        return result[0] if result else None
                    else:
                        error_text = await response.text()
                        print(f"❌ Failed to save company data: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            print(f"❌ Database save error: {e}")
            return None

    async def get_company_by_name(self, company_name: str) -> Optional[Dict[str, Any]]:
        """
        Get company data by name from Supabase
        
        Args:
            company_name: Name of the company to search for
            
        Returns:
            Company data or None if not found
        """
        if not self.available:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.supabase_url}/rest/v1/company_enrichment"
                headers = {
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                    "Content-Type": "application/json"
                }
                
                # Search for company by name (case insensitive)
                params = {
                    "company_name": f"ilike.{company_name}"
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result[0] if result else None
                    else:
                        print(f"❌ Failed to fetch company: {response.status}")
                        return None
                        
        except Exception as e:
            print(f"❌ Database fetch error: {e}")
            return None

    async def save_crawl_data(self, crawl_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save crawl data to Supabase (legacy method for backward compatibility)
        
        Args:
            crawl_data: Dictionary containing crawl information
            
        Returns:
            Saved data or None if failed
        """
        # Convert crawl_data to company_data format and save
        company_data = {
            "company_name": crawl_data.get("company_name", ""),
            "description": crawl_data.get("description"),
            "products": crawl_data.get("products"),
            "location": crawl_data.get("location"),
            "industry": crawl_data.get("industry"),
            "num_of_emp": crawl_data.get("num_of_emp"),
            "revenue": crawl_data.get("revenue"),
            "market_share": crawl_data.get("market_share")
        }
        
        return await self.save_company_data(company_data)
    
    async def delete_image_from_storage(self, image_url: str) -> bool:
        """
        Delete an image from Supabase Storage
        
        Args:
            image_url: The public URL of the image to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.available or not image_url:
            return False
        
        try:
            # Extract filename from URL
            # URL format: https://project.supabase.co/storage/v1/object/public/business-card-images/scanned-cards/filename.jpg
            if "/business-card-images/scanned-cards/" in image_url:
                filename = image_url.split("/business-card-images/scanned-cards/")[-1]
                
                # Delete from Supabase Storage
                storage_url = f"{self.supabase_url}/storage/v1/object/business-card-images/scanned-cards/{filename}"
                headers = {
                    "Authorization": f"Bearer {self.supabase_key}"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.delete(storage_url, headers=headers) as response:
                        if response.status == 200:
                            print(f"✅ Image deleted from storage: {filename}")
                            return True
                        else:
                            error_text = await response.text()
                            print(f"❌ Failed to delete image: {response.status} - {error_text}")
                            return False
            else:
                print(f"❌ Invalid image URL format: {image_url}")
                return False
                
        except Exception as e:
            print(f"❌ Image deletion error: {e}")
            return False
    
    async def get_scanned_cards(self) -> Optional[list]:
        """
        Fetch all records from the 'scanned_cards' table in the 'research-tables' database.
        """
        url = f"{self.supabase_url}/rest/v1/scanned_cards"
        headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
            }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Retrieved scanned cards:")
                    return data
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to fetch scanned cards: {response.status} - {error_text}")
                    return None
# Global instance
database_service = DatabaseService()
