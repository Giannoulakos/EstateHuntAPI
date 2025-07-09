"""
Database service for PropertyMatch AI API
Provides Prisma client connection and database utilities
"""

import os
from typing import Optional
from prisma import Prisma
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseService:
    """Service class for database operations using Prisma"""
    
    def __init__(self):
        self.client: Optional[Prisma] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to the database"""
        if self._connected:
            return
            
        try:
            self.client = Prisma()
            await self.client.connect()
            self._connected = True
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from the database"""
        if self.client and self._connected:
            await self.client.disconnect()
            self._connected = False
            logger.info("Database connection closed")
    
    def get_client(self) -> Prisma:
        """Get the Prisma client instance"""
        if not self.client or not self._connected:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.client
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._connected and self.client is not None

# Global database service instance
db_service = DatabaseService()

async def get_prisma() -> Prisma:
    """
    Dependency function to get Prisma client for FastAPI endpoints
    
    Returns:
        Prisma: Connected Prisma client instance
        
    Raises:
        RuntimeError: If database is not connected
    """
    if not db_service.is_connected:
        await db_service.connect()
    
    return db_service.get_client()

@asynccontextmanager
async def get_db_context():
    """
    Context manager for database operations
    Ensures proper connection and cleanup
    
    Usage:
        async with get_db_context() as db:
            # Use db (Prisma client) here
            result = await db.user.find_many()
    """
    try:
        if not db_service.is_connected:
            await db_service.connect()
        yield db_service.get_client()
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise
    finally:
        # Keep connection alive for performance
        # Only disconnect on app shutdown
        pass

async def init_database():
    """Initialize database connection on app startup"""
    try:
        await db_service.connect()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def close_database():
    """Close database connection on app shutdown"""
    try:
        await db_service.disconnect()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

# Health check function
async def check_database_health() -> dict:
    """
    Check database connectivity and health
    
    Returns:
        dict: Health status information
    """
    try:
        if not db_service.is_connected:
            await db_service.connect()
        
        # Simple query to test connection
        client = db_service.get_client()
        
        # You can add a simple query here to test the connection
        # For now, just check if client exists
        status = {
            "status": "healthy",
            "connected": db_service.is_connected,
            "database": "mongodb",
            "timestamp": str(datetime.now())
        }
        
        return status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
            "timestamp": str(datetime.now())
        }

# Import datetime for health check
from datetime import datetime
