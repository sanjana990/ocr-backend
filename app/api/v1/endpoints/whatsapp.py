#!/usr/bin/env python3
"""
WhatsApp integration endpoints
Handles incoming WhatsApp messages and business card processing
"""

from fastapi import APIRouter, Request, HTTPException, Form
from typing import Dict, Any, Optional
import structlog
import os

from app.services.twilio_whatsapp_service import twilio_whatsapp_service
from app.services.ocr_service import OCRService

# Import Apollo service for company enrichment
try:
    from apollo_service import apollo_service
    APOLLO_AVAILABLE = apollo_service.available
except ImportError:
    APOLLO_AVAILABLE = False

router = APIRouter()
logger = structlog.get_logger(__name__)

# Initialize OCR service
ocr_service = OCRService()


@router.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),
    To: str = Form(...),
    Body: str = Form(None),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None)
):
    """Handle incoming WhatsApp messages via Twilio"""
    try:
        # Extract phone number (remove whatsapp: prefix)
        sender_number = From.replace("whatsapp:", "")
        
        logger.info("📱 WhatsApp message received", 
                   from_number=sender_number, 
                   has_media=NumMedia != "0",
                   body=Body)
        
        # Handle text messages
        if Body and not NumMedia:
            return await handle_text_message(Body, sender_number)
        
        # Handle image messages
        if NumMedia and int(NumMedia) > 0 and MediaUrl0:
            return await handle_image_message(MediaUrl0, MediaContentType0, sender_number)
        
        # Default response
        return twilio_whatsapp_service.create_webhook_response(
            "📸 Please send a photo of a business card to get started!"
        )
        
    except Exception as e:
        logger.error("❌ WhatsApp webhook error", error=str(e))
        return twilio_whatsapp_service.create_webhook_response(
            "❌ An error occurred. Please try again."
        )


async def handle_text_message(body: str, sender_number: str) -> str:
    """Handle incoming text messages"""
    body_lower = body.lower().strip()
    
    # Welcome messages
    if body_lower in ["hi", "hello", "start", "help"]:
        welcome_message = (
            "👋 *Welcome to Business Card Scanner!*\n\n"
            "📸 Send me a photo of a business card and I'll extract all the information for you!\n\n"
            "✨ *Features:*\n"
            "• Extract contact details (name, title, company)\n"
            "• Scan phone numbers and emails\n"
            "• Detect QR codes and websites\n"
            "• Company data enrichment\n"
            "• Industry analysis\n\n"
            "Just send a clear photo of any business card to get started! 🚀"
        )
        return twilio_whatsapp_service.create_webhook_response(welcome_message)
    
    # Help message
    elif body_lower in ["help", "commands", "?"]:
        help_message = (
            "📸 *How to use Business Card Scanner:*\n\n"
            "1. Take a clear photo of a business card\n"
            "2. Send the photo to this chat\n"
            "3. Get instant analysis with:\n"
            "   • Contact information\n"
            "   • QR code data\n"
            "   • Company enrichment\n"
            "   • Industry details\n\n"
            "💡 *Tips for best results:*\n"
            "• Ensure good lighting\n"
            "• Keep the card flat\n"
            "• Avoid shadows and glare\n"
            "• Make sure text is readable"
        )
        return twilio_whatsapp_service.create_webhook_response(help_message)
    
    # Default text response
    else:
        help_message = (
            "📸 Please send a photo of a business card to get started!\n\n"
            "I can extract:\n"
            "• Names, titles, companies\n"
            "• Phone numbers and emails\n"
            "• QR codes and websites\n"
            "• Company enrichment data\n\n"
            "Type 'help' for more information."
        )
        return twilio_whatsapp_service.create_webhook_response(help_message)


async def handle_image_message(media_url: str, content_type: str, sender_number: str) -> str:
    """Handle incoming image messages"""
    try:
        # Validate content type
        if not content_type or not content_type.startswith('image/'):
            return twilio_whatsapp_service.create_webhook_response(
                "❌ Please send a valid image file (JPG, PNG, etc.)"
            )
        
        logger.info("📷 Processing business card image", 
                   media_url=media_url, 
                   content_type=content_type)
        
        # Download image from Twilio
        image_data = await twilio_whatsapp_service.download_media(media_url)
        if not image_data:
            return twilio_whatsapp_service.create_webhook_response(
                "❌ Failed to download image. Please try again."
            )
        
        logger.info("📥 Image downloaded", size=len(image_data))
        
        # Process with OCR service
        result = await ocr_service.extract_business_card_data(image_data, use_vision=True)
        
        if not result["success"]:
            error_message = (
                "❌ Sorry, I couldn't process the image.\n\n"
                "💡 *Tips for better results:*\n"
                "• Ensure good lighting\n"
                "• Keep the card flat and straight\n"
                "• Avoid shadows and glare\n"
                "• Make sure all text is visible\n\n"
                "Please try with a clearer photo!"
            )
            return twilio_whatsapp_service.create_webhook_response(error_message)
        
        logger.info("✅ OCR processing completed", 
                   confidence=result.get("confidence", 0),
                   qr_count=result.get("qr_count", 0))
        
        # Get company enrichment if company found
        company_name = result.get("data", {}).get("company")
        if company_name and APOLLO_AVAILABLE:
            try:
                logger.info("🏢 Starting company enrichment", company=company_name)
                enrichment = await apollo_service.search_company(company_name)
                result["enrichment"] = enrichment
                logger.info("✅ Company enrichment completed", company=company_name)
            except Exception as e:
                logger.warning("⚠️ Company enrichment failed", company=company_name, error=str(e))
        elif company_name and not APOLLO_AVAILABLE:
            logger.info("⚠️ Apollo.io not available, skipping enrichment")
        
        # Send formatted response via WhatsApp API (async)
        try:
            await twilio_whatsapp_service.send_business_card_analysis(sender_number, result)
            logger.info("📤 Business card analysis sent", to=sender_number)
        except Exception as e:
            logger.error("❌ Failed to send analysis", error=str(e))
            return twilio_whatsapp_service.create_webhook_response(
                "❌ Analysis completed but failed to send results. Please try again."
            )
        
        # Return empty response (we sent message via API)
        return twilio_whatsapp_service.create_webhook_response("")
        
    except Exception as e:
        logger.error("❌ Error processing image", error=str(e))
        return twilio_whatsapp_service.create_webhook_response(
            "❌ Error processing image. Please try again with a different photo."
        )


@router.get("/status")
async def whatsapp_status():
    """Check WhatsApp service status"""
    return {
        "service_available": twilio_whatsapp_service.available,
        "apollo_available": APOLLO_AVAILABLE,
        "ocr_engines": list(ocr_service.engines.keys())
    }
