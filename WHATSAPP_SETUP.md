# WhatsApp Business Card Scanner Setup Guide

## Overview
This guide will help you set up a WhatsApp bot that can scan business cards and return formatted contact information with data enrichment.

## Features
- 📸 **Business Card OCR**: Extract contact details from images
- 📱 **QR Code Detection**: Scan QR codes on business cards
- 🏢 **Company Enrichment**: Get company data via Apollo.io
- 💬 **WhatsApp Integration**: Send results directly via WhatsApp
- 🎯 **High Accuracy**: Multiple OCR engines with confidence scoring

## Setup Steps

### 1. Install Dependencies
```bash
# Install Twilio
pip install twilio>=8.0.0

# Or install all requirements
pip install -r requirements.txt
```

### 2. Twilio Account Setup

#### A. Create Twilio Account
1. Go to [Twilio Console](https://console.twilio.com/)
2. Sign up for a free account
3. Verify your phone number

#### B. Enable WhatsApp Sandbox
1. In Twilio Console, go to **Messaging** → **Try it out** → **Send a WhatsApp message**
2. Follow the instructions to connect your WhatsApp
3. You'll get a sandbox number like `+14155238886`
4. Send `join <sandbox-code>` to this number from your WhatsApp

#### C. Get Credentials
1. Go to **Account** → **API keys & tokens**
2. Copy your **Account SID** and **Auth Token**
3. Note your WhatsApp sandbox number

### 3. Environment Configuration

#### A. Update .env file
```bash
# Copy environment template
cp env.example .env

# Edit .env with your Twilio credentials
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
```

#### B. Optional: Apollo.io for Company Enrichment
```bash
# Add to .env for company data enrichment
APOLLO_API_KEY=your_apollo_api_key_here
```

### 4. Deploy Your Application

#### A. Local Development
```bash
# Start your FastAPI server
python run.py

# Your WhatsApp webhook will be available at:
# http://localhost:8000/api/v1/whatsapp/webhook
```

#### B. Production Deployment (Render)
1. Deploy your app to Render
2. Note your production URL: `https://your-app.onrender.com`
3. Set webhook URL in Twilio Console

### 5. Configure Twilio Webhook

#### A. Set Webhook URL
1. In Twilio Console, go to **Phone Numbers** → **Manage** → **WhatsApp Sandbox**
2. Set webhook URL: `https://your-domain.com/api/v1/whatsapp/webhook`
3. Set HTTP method: `POST`
4. Save configuration

#### B. Test Webhook
1. Send a test message to your WhatsApp sandbox
2. Check your server logs for incoming messages
3. Verify webhook is receiving messages

### 6. Testing Your Bot

#### A. Basic Test
1. Send `hi` to your WhatsApp sandbox number
2. You should receive a welcome message
3. Send a business card photo
4. Get formatted analysis

#### B. Advanced Testing
1. Test with different business card types
2. Test with QR codes
3. Test with poor quality images
4. Verify company enrichment (if Apollo.io configured)

## Usage Examples

### User Journey
1. **User sends**: "hi" → **Bot responds**: Welcome message
2. **User sends**: Business card photo → **Bot responds**: Formatted analysis
3. **User sends**: "help" → **Bot responds**: Usage instructions

### Sample Response
```
📇 Business Card Analysis

👤 Name: John Smith
💼 Title: Sales Director
🏢 Company: TechCorp Solutions
📞 Phone: +1-555-0123
📧 Email: john@techcorp.com
🌐 Website: www.techcorp.com

📱 QR Codes Found: 1
   1. linkedin.com/in/johnsmith

🏢 Company Enrichment:
🏭 Industry: Technology
👥 Company Size: 51-200 employees
📍 HQ Location: San Francisco, CA
📝 Description: Leading provider of enterprise software solutions...
🎯 Confidence: 94.2%
⚙️ Processed with: Tesseract
```

## API Endpoints

### WhatsApp Webhook
- **URL**: `/api/v1/whatsapp/webhook`
- **Method**: `POST`
- **Purpose**: Handle incoming WhatsApp messages

### Status Check
- **URL**: `/api/v1/whatsapp/status`
- **Method**: `GET`
- **Purpose**: Check service availability

## Troubleshooting

### Common Issues

#### 1. Webhook Not Receiving Messages
- Check webhook URL is correct
- Verify HTTPS is enabled
- Check server logs for errors

#### 2. Images Not Processing
- Verify image format (JPG, PNG supported)
- Check image size (max 10MB)
- Ensure good image quality

#### 3. Company Enrichment Not Working
- Check Apollo.io API key
- Verify company name extraction
- Check API rate limits

### Debug Commands
```bash
# Check service status
curl https://your-domain.com/api/v1/whatsapp/status

# Test webhook locally
curl -X POST http://localhost:8000/api/v1/whatsapp/webhook \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "From=whatsapp:+1234567890&Body=hi"
```

## Production Considerations

### 1. Rate Limits
- Twilio: 1 message per second per number
- Apollo.io: Check your plan limits
- OCR processing: Consider queue system for high volume

### 2. Error Handling
- Implement retry logic for failed messages
- Log all processing errors
- Graceful degradation when services unavailable

### 3. Security
- Validate webhook signatures
- Implement rate limiting
- Secure API keys and credentials

### 4. Monitoring
- Set up logging for all messages
- Monitor processing times
- Track success/failure rates

## Cost Estimation

### Twilio WhatsApp
- **Sandbox**: Free (limited to 24 hours)
- **Production**: ~$0.005 per message
- **Monthly**: ~$15-50 for moderate usage

### Apollo.io (Optional)
- **Free tier**: 50 requests/month
- **Paid plans**: $39+/month for higher limits

### Total Monthly Cost
- **Basic setup**: $0-15 (Twilio sandbox + free Apollo)
- **Production**: $15-65 (depending on usage)

## Next Steps

1. **Test thoroughly** with various business cards
2. **Monitor performance** and optimize
3. **Add more features** (contact saving, CRM integration)
4. **Scale up** to production WhatsApp Business API
5. **Add analytics** and usage tracking

## Support

For issues or questions:
1. Check server logs
2. Verify webhook configuration
3. Test with simple messages first
4. Check Twilio Console for message status


