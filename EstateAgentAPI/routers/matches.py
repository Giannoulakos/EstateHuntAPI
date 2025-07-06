from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from agent_rag import find_matching_customers_api

router = APIRouter(
    prefix="/matches",
    tags=["matches"],
    responses={404: {"description": "Not found"}}
)

class PropertyQuery(BaseModel):
    property_description: str
    user_id: str
    url: str
    k: Optional[int] = 3

@router.post("/find")
def find_customer_matches(query: PropertyQuery):
    """
    Find customers who might be interested in a property for a specific user.
    
    Args:
        query: PropertyQuery containing property description, user_id, and number of matches
    
    Returns:
        JSON response with matching customers and AI analysis
    """
    try:
        result = find_matching_customers_api(
            property_query=query.property_description,
            user_id=query.user_id,
            url=query.url,
            k=query.k
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/test")
def test_match():
    """Test endpoint with a sample property query"""
    sample_query = """
    Luxury 2-bedroom apartment in downtown Manhattan with modern amenities.
    Price: $950,000
    Features: Gym, pool, doorman, parking, city views
    Description: Modern high-rise building with luxury finishes and prime location.
    """
    
    result = find_matching_customers_api(
        property_query=sample_query, 
        user_id="test_user_1",
        url="sample_customers.json", 
        k=3
    )
    return result

@router.get("/stats")
def get_match_stats():
    """Get statistics about matches performed"""
    return {
        "message": "Match statistics endpoint",
        "total_queries": 0,
        "active_users": 0,
        "avg_response_time": "0ms"
    }
