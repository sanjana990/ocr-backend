"""
Visitor database model
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Visitor(Base):
    """Visitor model"""
    __tablename__ = "visitors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True)
    company = Column(String(255))
    phone = Column(String(50))
    title = Column(String(255))
    summary = Column(Text)
    confidence_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    company_id = Column(Integer, ForeignKey("companies.id"))
    company_rel = relationship("Company", back_populates="visitors")
    
    interactions = relationship("Interaction", back_populates="visitor")
    enrichment_logs = relationship("EnrichmentLog", back_populates="visitor")
