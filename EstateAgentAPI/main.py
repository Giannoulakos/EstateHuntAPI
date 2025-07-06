from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import matches

app = FastAPI(
    title="PropertyMatch AI", 
    description="Real Estate Customer Matching API with AI-powered analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
            "system": "/system"
        }
    }
