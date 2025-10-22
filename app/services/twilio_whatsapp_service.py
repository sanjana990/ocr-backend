#!/usr/bin/env python3
"""
Twilio WhatsApp Service
Handles WhatsApp messaging via Twilio for business card scanning
"""

import os
import aiohttp
import structlog
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from typing import Dict, Any, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = structlog.get_logger(__name__)


class TwilioWhatsAppService:
    """Twilio WhatsApp integration for business card scanning"""
    
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")  # e.g., "whatsapp:+14155238886"
        
        if not all([self.account_sid, self.auth_token, self.whatsapp_number]):
            logger.warning("Twilio credentials not configured - WhatsApp service will not be available")
            self.available = False
        else:
            self.available = True
            self.client = Client(self.account_sid, self.auth_token)
            logger.info("✅ Twilio WhatsApp service initialized")
    
    async def send_message(self, to: str, message: str) -> Dict[str, Any]:
        """Send WhatsApp message via Twilio"""
        if not self.available:
            return {"success": False, "error": "Twilio not configured"}
        
        try:
            message_obj = self.client.messages.create(
                body=message,
                from_=self.whatsapp_number,
                to=f"whatsapp:{to}"
            )
            logger.info("📱 WhatsApp message sent", to=to, message_sid=message_obj.sid)
            return {"success": True, "message_sid": message_obj.sid}
        except Exception as e:
            logger.error("❌ Failed to send WhatsApp message", to=to, error=str(e))
            return {"success": False, "error": str(e)}
    
    async def send_business_card_analysis(self, to: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Send formatted business card analysis"""
        message = self._format_business_card_message(analysis_result)
        return await self.send_message(to, message)
    
    def _format_business_card_message(self, result: Dict[str, Any]) -> str:
        """Format business card data into WhatsApp message"""
        data = result.get("data", {})
        enrichment = result.get("enrichment", {})
        qr_codes = result.get("qr_codes", [])
        
        message = "📇 *Business Card Analysis*\n\n"
        
        # Basic contact info
        if data.get("name"):
            message += f"👤 *Name:* {data['name']}\n"
        if data.get("title"):
            message += f"💼 *Title:* {data['title']}\n"
        if data.get("company"):
            message += f"🏢 *Company:* {data['company']}\n"
        if data.get("phone"):
            message += f"📞 *Phone:* {data['phone']}\n"
        if data.get("email"):
            message += f"📧 *Email:* {data['email']}\n"
        if data.get("website"):
            message += f"🌐 *Website:* {data['website']}\n"
        if data.get("address"):
            message += f"📍 *Address:* {data['address']}\n"
        
        # QR codes found
        if qr_codes:
            message += f"\n📱 *QR Codes Found:* {len(qr_codes)}\n"
            for i, qr in enumerate(qr_codes[:3], 1):  # Show max 3 QR codes
                qr_data = qr.get('data', '')
                if len(qr_data) > 50:
                    qr_data = qr_data[:50] + "..."
                message += f"   {i}. {qr_data}\n"
        
        # Enriched company data
        if enrichment:
            message += f"\n🏢 *Company Enrichment:*\n"
            if enrichment.get("industry"):
                message += f"🏭 *Industry:* {enrichment['industry']}\n"
            if enrichment.get("size"):
                message += f"👥 *Company Size:* {enrichment['size']}\n"
            if enrichment.get("hq_location"):
                message += f"📍 *HQ Location:* {enrichment['hq_location']}\n"
            if enrichment.get("description"):
                desc = enrichment['description'][:100]
                message += f"📝 *Description:* {desc}...\n"
            if enrichment.get("linkedin_url"):
                message += f"🔗 *LinkedIn:* {enrichment['linkedin_url']}\n"
        
        # Add confidence score
        confidence = result.get("confidence", 0)
        if confidence > 0:
            message += f"\n🎯 *Confidence:* {confidence:.1f}%"
        
        # Add processing info
        engine_used = result.get("engine_used", "unknown")
        message += f"\n⚙️ *Processed with:* {engine_used.title()}"
        
        return message
    
    def create_webhook_response(self, message: str) -> str:
        """Create TwiML response for webhook"""
        response = MessagingResponse()
        response.message(message)
        return str(response)
    
    async def download_media(self, media_url: str) -> Optional[bytes]:
        """Download media from Twilio with authentication"""
        try:
            # Twilio media URLs require authentication
            auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(media_url, auth=auth) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error("Failed to download media", 
                                   status=response.status, 
                                   url=media_url)
                        return None
        except Exception as e:
            logger.error("Error downloading media", error=str(e), url=media_url)
            return None


# Global instance
twilio_whatsapp_service = TwilioWhatsAppService()
