"""
Visitor Pydantic schemas
"""

from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class VisitorBase(BaseModel):
    """Base visitor schema"""
    name: str
    email: Optional[EmailStr] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    title: Optional[str] = None


class VisitorCreate(VisitorBase):
    """Visitor creation schema"""
    pass


class VisitorUpdate(BaseModel):
    """Visitor update schema"""
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None


class VisitorResponse(VisitorBase):
    """Visitor response schema"""
    id: int
    summary: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
