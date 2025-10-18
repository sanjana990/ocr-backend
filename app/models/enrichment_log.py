"""
Enrichment log database model
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class EnrichmentLog(Base):
    """Enrichment log model"""
    __tablename__ = "enrichment_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    visitor_id = Column(Integer, ForeignKey("visitors.id"))
    source = Column(String(100), nullable=False)  # 'ocr', 'web', 'social', etc.
    data = Column(JSON)
    status = Column(String(50), default="pending")  # 'pending', 'completed', 'failed'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    visitor = relationship("Visitor", back_populates="enrichment_logs")
