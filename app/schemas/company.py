"""
Company Pydantic schemas
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class CompanyBase(BaseModel):
    """Base company schema"""
    name: str
    domain: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None


class CompanyCreate(CompanyBase):
    """Company creation schema"""
    funding_info: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class CompanyResponse(CompanyBase):
    """Company response schema"""
    id: int
    funding_info: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
