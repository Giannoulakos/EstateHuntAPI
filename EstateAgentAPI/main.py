from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routers import matches, uploads
from database import init_database, close_database, check_database_health

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    await init_database()
    yield
    # Shutdown
    await close_database()

app = FastAPI(
    title="PropertyMatch AI", 
    description="Real Estate Customer Matching API with AI-powered analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(matches.router)
app.include_router(uploads.router)

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "PropertyMatch AI - Real Estate Customer Matching API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "authentication": "/auth",
            "customer_matching": "/matches", 
            "file_uploads": "/uploads",
            "system": "/system"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_health = await check_database_health()
    return {
        "status": "ok",
        "database": db_health
    }
