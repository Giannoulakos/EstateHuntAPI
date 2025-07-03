from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from agent_rag import find_matching_customers_api

app = FastAPI(title="PropertyMatch AI", description="Real Estate Customer Matching API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)

class PropertyQuery(BaseModel):
    property_description: str
    k: Optional[int] = 3

@app.get("/")
def read_root():
    return {"message": "PropertyMatch AI - Real Estate Customer Matching API"}

@app.post("/find-matches")
def find_customer_matches(query: PropertyQuery):
    """
    Find customers who might be interested in a property.
    
    Args:
        query: PropertyQuery containing property description and number of matches
    
    Returns:
        JSON response with matching customers and AI analysis
    """
    try:
        result = find_matching_customers_api(
            property_query=query.property_description,
            k=query.k
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/test-match")
def test_match():
    """Test endpoint with a sample property query"""
    sample_query = """
    Luxury 2-bedroom apartment in downtown Manhattan with modern amenities.
    Price: $950,000
    Features: Gym, pool, doorman, parking, city views
    Description: Modern high-rise building with luxury finishes and prime location.
    """
    
    result = find_matching_customers_api(sample_query, k=3)
    return result
