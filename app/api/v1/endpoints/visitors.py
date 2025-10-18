"""
Visitor management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.visitor import Visitor
from app.schemas.visitor import VisitorCreate, VisitorResponse, VisitorUpdate

router = APIRouter()


@router.get("/", response_model=List[VisitorResponse])
async def get_visitors(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all visitors"""
    # TODO: Implement visitor retrieval
    return []


@router.get("/{visitor_id}", response_model=VisitorResponse)
async def get_visitor(
    visitor_id: int,
    db: Session = Depends(get_db)
):
    """Get visitor by ID"""
    # TODO: Implement visitor retrieval by ID
    raise HTTPException(status_code=404, detail="Visitor not found")


@router.post("/", response_model=VisitorResponse)
async def create_visitor(
    visitor: VisitorCreate,
    db: Session = Depends(get_db)
):
    """Create new visitor"""
    # TODO: Implement visitor creation
    raise HTTPException(status_code=501, detail="Not implemented")


@router.put("/{visitor_id}", response_model=VisitorResponse)
async def update_visitor(
    visitor_id: int,
    visitor: VisitorUpdate,
    db: Session = Depends(get_db)
):
    """Update visitor"""
    # TODO: Implement visitor update
    raise HTTPException(status_code=501, detail="Not implemented")


@router.delete("/{visitor_id}")
async def delete_visitor(
    visitor_id: int,
    db: Session = Depends(get_db)
):
    """Delete visitor"""
    # TODO: Implement visitor deletion
    raise HTTPException(status_code=501, detail="Not implemented")
