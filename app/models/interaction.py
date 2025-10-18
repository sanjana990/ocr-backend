"""
Interaction database model
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Interaction(Base):
    """Interaction model"""
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    visitor_id = Column(Integer, ForeignKey("visitors.id"))
    type = Column(String(50), nullable=False)  # 'chat', 'email', 'meeting', etc.
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    visitor = relationship("Visitor", back_populates="interactions")
