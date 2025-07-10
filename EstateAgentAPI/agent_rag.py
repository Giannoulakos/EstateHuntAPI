from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.documents import Document
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
import pandas as pd

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure for real estate customer data
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", ", ", " ", ""]
)

# Initialize Groq Chat model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=1000
)

# Create embeddings using SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')

class RealEstateRAGAgent:
    def __init__(self):
        self.llm = llm
        self.encoder = encoder
        self.qdrant_client = None
        self.customers_data = []
        self.collection_name = "customers"
    
    def _init_qdrant(self):
        """Initialize Qdrant client if not already initialized"""
        if not self.qdrant_client:
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=60
            )
    
    def load_customers_from_csv(self, csv_url: str, user_id: str):
        """Load customers from CSV file with headers: Contact ID,First Name,Last Name,Email,Phone,Primary Address,Type"""
        self._init_qdrant()
        
        # Read CSV into DataFrame
        df = pd.read_csv(csv_url)
        
        # Convert DataFrame to list of dictionaries
        self.customers_data = df.to_dict(orient='records')
        
        # Convert customers to documents for embedding
        documents = []
        for customer in self.customers_data:
            customer_text = self._customer_to_text(customer)
            
            # Create Document object with metadata for CSV structure
            doc = Document(
                page_content=customer_text,
                metadata={
                    "user_id": user_id,
                    "customer_id": customer.get("Contact ID", ""),
                    "customer_data": customer,
                    "name": f"{customer.get('First Name', '')} {customer.get('Last Name', '')}".strip(),
                    "email": customer.get("Email", ""),
                    "phone": customer.get("Phone", ""),
                    "address": customer.get("Primary Address", ""),
                    "type": customer.get("Type", ""),
                    # For CSV, these might be empty or need to be mapped from other columns
                    "property_type": customer.get("property_type", ""),
                    "location": customer.get("location", customer.get("Primary Address", "")),
                    "min_price": customer.get("min_price", 0),
                    "max_price": customer.get("max_price", 0),
                    "bedrooms": customer.get("bedrooms", 0),
                    "bathrooms": customer.get("bathrooms", 0)
                }
            )
            documents.append(doc)
        
        # Update collection if it exists, otherwise create new one
        try:
            # Try to get collection info to see if it exists
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            # Get the current count to generate unique IDs
            current_count = collection_info.points_count
            
            # If exists, add new points with unique IDs
            points = []
            for idx, doc in enumerate(documents):
                vector = self.encoder.encode(doc.page_content).tolist()
                point = models.PointStruct(
                    id=current_count + idx,  # Use current count + index for unique IDs
                    vector=vector,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )
                points.append(point)
            
            self.qdrant_client.upload_points(
                collection_name=self.collection_name,
                points=points
            )
        except:
            # Collection doesn't exist, create new one
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=models.Distance.COSINE
                )
            )

            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.user_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            
            # Upload points
            points = []
            for idx, doc in enumerate(documents):
                vector = self.encoder.encode(doc.page_content).tolist()
                point = models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )
                points.append(point)
            
            self.qdrant_client.upload_points(
                collection_name=self.collection_name,
                points=points
            )
        
        print(f"Loaded {len(self.customers_data)} customers from CSV into Qdrant vector store")
    
    def load_customers_from_json(self, json_file_path: str, user_id: str):
        """Load customers from JSON file"""
        self._init_qdrant()
        
        with open(json_file_path, 'r') as file:
            self.customers_data = json.load(file)
        
        # Convert customers to documents for embedding
        documents = []
        for customer in self.customers_data:
            customer_text = self._customer_to_text(customer)
            
            # Create Document object with metadata for JSON structure
            doc = Document(
                page_content=customer_text,
                metadata={
                    "user_id": user_id,
                    "customer_id": customer.get("id", ""),
                    "customer_data": customer,
                    "name": customer.get("name", ""),
                    "email": customer.get("email", ""),
                    "phone": customer.get("phone", ""),
                    "property_type": customer.get("preferences", {}).get("property_type", ""),
                    "location": customer.get("preferences", {}).get("location", ""),
                    "min_price": customer.get("preferences", {}).get("price_range", {}).get("min", 0),
                    "max_price": customer.get("preferences", {}).get("price_range", {}).get("max", 0),
                    "bedrooms": customer.get("preferences", {}).get("bedrooms", 0),
                    "bathrooms": customer.get("preferences", {}).get("bathrooms", 0)
                }
            )
            documents.append(doc)
        
        # Update collection if it exists, otherwise create new one
        try:
            # Try to get collection info to see if it exists
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            # Get the current count to generate unique IDs
            current_count = collection_info.points_count
            
            # If exists, add new points with unique IDs
            points = []
            for idx, doc in enumerate(documents):
                vector = self.encoder.encode(doc.page_content).tolist()
                point = models.PointStruct(
                    id=current_count + idx,  # Use current count + index for unique IDs
                    vector=vector,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )
                points.append(point)
            
            self.qdrant_client.upload_points(
                collection_name=self.collection_name,
                points=points
            )
        except:
            # Collection doesn't exist, create new one
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=models.Distance.COSINE
                )
            )
            
            # Create index for user_id filtering
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.user_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Upload points
            points = []
            for idx, doc in enumerate(documents):
                vector = self.encoder.encode(doc.page_content).tolist()
                point = models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                )
                points.append(point)
            
            self.qdrant_client.upload_points(
                collection_name=self.collection_name,
                points=points
            )
        
        print(f"Loaded {len(self.customers_data)} customers from JSON into Qdrant vector store")
    
    def _customer_to_text(self, customer: dict) -> str:
        """Convert customer data to searchable text - handles both JSON and CSV formats"""
        if "preferences" in customer:
            # JSON format with nested preferences
            preferences = customer.get("preferences", {})
            text_parts = [
                f"Customer: {customer.get('name', '')}",
                f"Profile: {customer.get('profile', '')}",
                f"Property Type: {preferences.get('property_type', '')}",
                f"Location: {preferences.get('location', '')}",
                f"Budget: ${preferences.get('price_range', {}).get('min', 0):,} to ${preferences.get('price_range', {}).get('max', 0):,}",
                f"Bedrooms: {preferences.get('bedrooms', '')}",
                f"Bathrooms: {preferences.get('bathrooms', '')}",
                f"Amenities: {', '.join(preferences.get('amenities', []))}",
                f"Description: {preferences.get('description', '')}"
            ]
        else:
            # CSV format with flat structure
            text_parts = [
                f"Customer: {customer.get('First Name', '')} {customer.get('Last Name', '')}",
                f"Contact ID: {customer.get('Contact ID', '')}",
                f"Email: {customer.get('Email', '')}",
                f"Phone: {customer.get('Phone', '')}",
                f"Address: {customer.get('Primary Address', '')}",
                f"Type: {customer.get('Type', '')}",
                f"Profile: {customer.get('profile', '')}",
                f"Property Type: {customer.get('property_type', '')}",
                f"Location: {customer.get('location', '')}",
                f"Budget: {customer.get('budget', '')}",
                f"Bedrooms: {customer.get('bedrooms', '')}",
                f"Bathrooms: {customer.get('bathrooms', '')}",
                f"Amenities: {customer.get('amenities', '')}",
                f"Description: {customer.get('description', '')}"
            ]
        
        return " | ".join(filter(None, text_parts))  # Filter out empty strings
    
    def find_matching_customers(self, property_query: str, user_id: str, k: int = 3) -> List[Dict]:
        """Find customers matching a property description for a specific user"""
        if not self.qdrant_client:
            raise ValueError("No customers loaded. Call load_customers_from_json() or load_customers_from_csv() first.")
        
        # Encode the query
        query_vector = self.encoder.encode(property_query).tolist()
        
        # Search for similar customers with user_id filter applied at query level
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=k  # Now we can use the exact k since filtering is done at query level
        )
        
        matches = []
        for scored_point in search_result:
            payload = scored_point.payload
            customer_data = payload["metadata"]["customer_data"]
            matches.append({
                "customer": customer_data,
                "similarity_score": scored_point.score,
                "matching_text": payload["page_content"]
            })
        
        return matches
        
    
    def generate_personalized_pitch(self, property_description: str, customer_data: dict) -> str:
        """Generate a personalized sales pitch using Groq"""
        # Handle both JSON and CSV formats
        if "preferences" in customer_data:
            # JSON format
            customer_name = customer_data.get("name", "Customer")
            preferences = customer_data.get("preferences", {})
        else:
            # CSV format
            customer_name = f"{customer_data.get('First Name', '')} {customer_data.get('Last Name', '')}".strip()
            preferences = {}
        
        system_prompt = """You are an expert real estate agent. Create a personalized, compelling sales pitch for a property to a specific customer. Make it professional, engaging, and tailored to their specific needs and preferences."""
        
        human_prompt = f"""
        Create a personalized sales pitch for this property to this customer:

        PROPERTY:
        {property_description}

        CUSTOMER:
        - Name: {customer_name}
        - Profile: {customer_data.get('profile', '')}
        - Email: {customer_data.get('Email', customer_data.get('email', ''))}
        - Phone: {customer_data.get('Phone', customer_data.get('phone', ''))}
        - Address: {customer_data.get('Primary Address', '')}
        - Type: {customer_data.get('Type', '')}
        - Budget: ${preferences.get('price_range', {}).get('min', 0):,} - ${preferences.get('price_range', {}).get('max', 0):,}
        - Preferred Type: {preferences.get('property_type', '')}
        - Preferred Location: {preferences.get('location', '')}
        - Desired Bedrooms: {preferences.get('bedrooms', '')}
        - Desired Bathrooms: {preferences.get('bathrooms', '')}
        - Required Amenities: {', '.join(preferences.get('amenities', []))}
        - Customer Notes: {preferences.get('description', '')}

        Write a compelling 2-3 paragraph pitch that:
        1. Addresses the customer by name
        2. Highlights how this property meets their specific needs
        3. Creates excitement without being pushy
        4. Mentions specific amenities they care about
        5. Connects to their lifestyle and profile

        Keep it professional, personalized, and persuasive.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def analyze_customer_match(self, property_description: str, customer_data: dict) -> str:
        """Analyze why a customer matches a property using Groq"""
        # Handle both JSON and CSV formats
        if "preferences" in customer_data:
            # JSON format
            customer_name = customer_data.get("name", "Customer")
            preferences = customer_data.get("preferences", {})
        else:
            # CSV format
            customer_name = f"{customer_data.get('First Name', '')} {customer_data.get('Last Name', '')}".strip()
            preferences = {}
        
        system_prompt = """You are a real estate analyst. Analyze why a customer might be interested in a property and provide specific, actionable insights."""
        
        human_prompt = f"""
        Analyze why this customer might be interested in this property:

        PROPERTY:
        {property_description}

        CUSTOMER:
        - Name: {customer_name}
        - Profile: {customer_data.get('profile', '')}
        - Email: {customer_data.get('Email', customer_data.get('email', ''))}
        - Phone: {customer_data.get('Phone', customer_data.get('phone', ''))}
        - Address: {customer_data.get('Primary Address', '')}
        - Type: {customer_data.get('Type', '')}
        - Budget: ${preferences.get('price_range', {}).get('min', 0):,} - ${preferences.get('price_range', {}).get('max', 0):,}
        - Preferred Type: {preferences.get('property_type', '')}
        - Preferred Location: {preferences.get('location', '')}
        - Desired Bedrooms: {preferences.get('bedrooms', '')}
        - Desired Bathrooms: {preferences.get('bathrooms', '')}
        - Required Amenities: {', '.join(preferences.get('amenities', []))}
        - Customer Notes: {preferences.get('description', '')}

        Provide 3-5 specific reasons why this customer would be interested, focusing on:
        1. Matching criteria (location, price, size, type)
        2. Lifestyle compatibility  
        3. Specific amenity matches
        4. Potential concerns or selling points

        Format as bullet points.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

# Initialize the RAG agent
rag_agent = RealEstateRAGAgent()

# API Function for FastAPI integration
def find_matching_customers_api(property_query: str, user_id: str, url: str = "sample_customers.csv", k: int = 3) -> Dict[str, Any]:
    """
    API function to find customers matching a property description for a specific user.
    
    Args:
        property_query (str): Description of the property to match against
        user_id (str): User ID to filter customers by
        url (str): URL to CSV or JSON file with customer data
        k (int): Number of top matches to return (default: 3)
    
    Returns:
        Dict containing matching customers with analysis
    """
    
    # Load customers from the provided URL
    try:
        if url.endswith('.csv'):
            rag_agent.load_customers_from_csv(url, user_id)
        elif url.endswith('.json'):
            rag_agent.load_customers_from_json(url, user_id)
        else:
            return {
                "success": False,
                "data": [],
                "message": "Invalid file format. Only .csv and .json files are supported.",
                "total_matches": 0
            }
    except Exception as e:
        return {
            "success": False,
            "data": [],
            "message": f"Error loading customer data: {str(e)}",
            "total_matches": 0
        }

    try:
        # Find matches using vector search with user_id filtering
        matches = rag_agent.find_matching_customers(property_query, user_id, k)
        
        if not matches:
            return {
                "success": True,
                "data": [],
                "message": "No matching customers found for this property and user.",
                "total_matches": 0
            }
        
        # Process matches and generate AI analysis
        results = []
        for match in matches:
            customer = match["customer"]
            
            # Handle both CSV and JSON formats for customer info
            if "preferences" in customer:
                # JSON format
                customer_name = customer.get("name", "")
                customer_email = customer.get("email", "")
                customer_phone = customer.get("phone", "")
                customer_id = customer.get("id", "")
            else:
                # CSV format
                customer_name = f"{customer.get('First Name', '')} {customer.get('Last Name', '')}".strip()
                customer_email = customer.get("Email", "")
                customer_phone = customer.get("Phone", "")
                customer_id = customer.get("Contact ID", "")
            
            # Prepare basic customer info
            customer_info = {
                "customer_id": customer_id,
                "customer_name": customer_name,
                "customer_email": customer_email,
                "customer_phone": customer_phone,
                "customer_profile": customer.get("profile", ""),
                "similarity_score": round(match["similarity_score"], 3),
                "preferences": customer.get("preferences", {}),
                "match_analysis": None,
                "personalized_pitch": None
            }
            
            # Generate AI analysis if Groq API is available
            if os.getenv("GROQ_API_KEY"):
                try:
                    customer_info["match_analysis"] = rag_agent.analyze_customer_match(
                        property_query, customer
                    )
                    customer_info["personalized_pitch"] = rag_agent.generate_personalized_pitch(
                        property_query, customer
                    )
                except Exception as e:
                    customer_info["match_analysis"] = f"AI analysis unavailable: {str(e)}"
                    customer_info["personalized_pitch"] = "Personalized pitch unavailable due to AI service error."
            else:
                customer_info["match_analysis"] = "AI analysis unavailable: GROQ_API_KEY not configured."
                customer_info["personalized_pitch"] = "Personalized pitch unavailable: GROQ_API_KEY not configured."
            
            results.append(customer_info)
        
        return {
            "success": True,
            "data": results,
            "message": f"Found {len(results)} matching customers successfully.",
            "total_matches": len(results)
        }
        
    except Exception as e:
        return {
            "success": False,
            "data": [],
            "message": f"Error finding matching customers: {str(e)}",
            "total_matches": 0
        }

# Test the system
if __name__ == "__main__":
    property_query = """
    Downtown office space for lease, 2000 sq ft, modern amenities
    """
    
    print("Testing CSV format...")
    result = find_matching_customers_api(property_query, "test_user_1", "sample_customers.csv", k=3)
    
    if result["success"]:
        print(f"Found {result['total_matches']} matches")
        for customer in result["data"]:
            print(f"- {customer['customer_name']} ({customer['similarity_score']})")
    else:
        print(f"Error: {result['message']}")
