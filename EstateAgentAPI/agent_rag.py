from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.documents import Document
import os
import json
from typing import List, Dict, Any

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
    model="llama-3.1-8b-instant",  # Updated to supported model
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=1000
)

# Create embeddings using HuggingFace (free alternative)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

class RealEstateRAGAgent:
    def __init__(self):
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = None
        self.customers_data = []
    
    def load_customers_from_json(self, json_file_path: str):
        """Load customers from JSON file"""
        with open(json_file_path, 'r') as file:
            self.customers_data = json.load(file)
        
        # Convert customers to documents for embedding
        documents = []
        for customer in self.customers_data:
            customer_text = self._customer_to_text(customer)
            
            # Create Document object with metadata
            from langchain_core.documents import Document
            doc = Document(
                page_content=customer_text,
                metadata={
                    "customer_id": customer["id"],
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
        
        # Create InMemoryVectorStore and add documents
        self.vectorstore = InMemoryVectorStore(embedding=self.embeddings)
        self.vectorstore.add_documents(documents)
        
        print(f"Loaded {len(self.customers_data)} customers into vector store")
    
    def _customer_to_text(self, customer: dict) -> str:
        """Convert customer data to searchable text"""
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
        
        return " | ".join(text_parts)
    
    def find_matching_customers(self, property_query: str, k: int = 3) -> List[Dict]:
        """Find customers matching a property description"""
        if not self.vectorstore:
            raise ValueError("No customers loaded. Call load_customers_from_json() first.")
        
        # Search for similar customers
        docs = self.vectorstore.similarity_search_with_score(property_query, k=k)
        
        matches = []
        for doc, score in docs:
            customer_data = doc.metadata["customer_data"]
            matches.append({
                "customer": customer_data,
                "similarity_score": score,
                "matching_text": doc.page_content
            })
        
        return matches
    
    def generate_personalized_pitch(self, property_description: str, customer_data: dict) -> str:
        """Generate a personalized sales pitch using Groq"""
        customer_name = customer_data.get("name", "Customer")
        preferences = customer_data.get("preferences", {})
        
        system_prompt = """You are an expert real estate agent. Create a personalized, compelling sales pitch for a property to a specific customer. Make it professional, engaging, and tailored to their specific needs and preferences."""
        
        human_prompt = f"""
        Create a personalized sales pitch for this property to this customer:

        PROPERTY:
        {property_description}

        CUSTOMER:
        - Name: {customer_name}
        - Profile: {customer_data.get('profile', '')}
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
        customer_name = customer_data.get("name", "Customer")
        preferences = customer_data.get("preferences", {})
        
        system_prompt = """You are a real estate analyst. Analyze why a customer might be interested in a property and provide specific, actionable insights."""
        
        human_prompt = f"""
        Analyze why this customer might be interested in this property:

        PROPERTY:
        {property_description}

        CUSTOMER:
        - Name: {customer_name}
        - Profile: {customer_data.get('profile', '')}
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

# Load sample customers (you need to create this file or use the existing one)
try:
    rag_agent.load_customers_from_json("/Users/giannisgiannoulakos/Documents/projects/RealEstateAgent/API/EstateAgentAPI/sample_customers.json")
except FileNotFoundError:
    print("json file not found. Please create it first.")

# API Function for FastAPI integration
def find_matching_customers_api(property_query: str, k: int = 3) -> Dict[str, Any]:
    """
    API function to find customers matching a property description.
    
    Args:
        property_query (str): Description of the property to match against
        k (int): Number of top matches to return (default: 3)
    
    Returns:
        Dict containing:
        - success (bool): Whether the operation was successful
        - data (List): List of matching customers with analysis
        - message (str): Status message
        - total_matches (int): Number of matches found
    """
    try:
        # Check if RAG agent is initialized
        if not rag_agent.vectorstore:
            return {
                "success": False,
                "data": [],
                "message": "RAG agent not initialized. Customer database not loaded.",
                "total_matches": 0
            }
        

        # Find matches using vector search
        matches = rag_agent.find_matching_customers(property_query, k)
        
        if not matches:
            return {
                "success": True,
                "data": [],
                "message": "No matching customers found for this property.",
                "total_matches": 0
            }
        
        # Process matches and generate AI analysis
        results = []
        for match in matches:
            customer = match["customer"]
            
            # Prepare basic customer info
            customer_info = {
                "customer_id": customer.get("id"),
                "customer_name": customer.get("name"),
                "customer_email": customer.get("email"),
                "customer_phone": customer.get("phone"),
                "customer_profile": customer.get("profile"),
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

# Legacy function for backward compatibility
def find_matching_customers(property_query: str, k: int = 3):
    """Find customers matching a property description (legacy function)"""
    result = find_matching_customers_api(property_query, k)
    return result.get("data", [])

# Test the system
if __name__ == "__main__":
    
    property_query = """
    historic townhouse in West Village
    """
    
    print("Searching for matching customers...")
    
    # Test basic vector search first
    if rag_agent.vectorstore:
        matches = rag_agent.find_matching_customers(property_query, k=3)
        print(f"Found {len(matches)} matches")
        
        for i, match in enumerate(matches, 1):
            customer = match["customer"]
            print(f"\n--- Match {i} ---")
            print(f"Customer: {customer.get('name')}")
            print(f"Email: {customer.get('email')}")
            print(f"Profile: {customer.get('profile')}")
            print(f"Similarity Score: {match['similarity_score']:.3f}")
            print(f"Matching Text: {match['matching_text'][:200]}...")
            
            # Only try Groq if GROQ_API_KEY is set
            if os.getenv("GROQ_API_KEY"):
                try:
                    analysis = rag_agent.analyze_customer_match(property_query, customer)
                    pitch = rag_agent.generate_personalized_pitch(property_query, customer)
                    print(f"\nMatch Analysis:\n{analysis}")
                    print(f"\nPersonalized Pitch:\n{pitch}")
                except Exception as e:
                    print(f"\nGroq analysis failed: {e}")
                    print("But vector search is working!")
            else:
                print("\nGroq API key not set - skipping AI analysis")
                print("Vector search is working correctly!")
            
            print("-" * 80)
    else:
        print("Vector store not initialized!")
