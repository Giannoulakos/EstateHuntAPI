from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import json
import os
import uuid
import base64
import io
from datetime import datetime
from database import DatabaseService, get_prisma
from agent_rag import RealEstateRAGAgent

router = APIRouter(
    prefix="/uploads",
    tags=["uploads"],
    responses={404: {"description": "Not found"}}
)

class UploadResponse(BaseModel):
    success: bool
    message: str
    file_id: str
    data_source_id: str
    record_count: int

class DataSourceInfo(BaseModel):
    id: str
    name: str
    description: Optional[str]
    file_type: str
    record_count: int
    created_at: str
    last_sync: Optional[str]

ALLOWED_EXTENSIONS = {".csv", ".json"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def get_user_id_from_auth0(authorization: Optional[str] = Header(None)) -> str:
    """
    Extract user ID from Auth0 token or use a default for demo purposes.
    In production, you would validate the Auth0 JWT token here.
    """
    if authorization and authorization.startswith("Bearer "):
        # In production, decode and validate the Auth0 JWT token
        # For now, return a demo user ID as a valid ObjectId
        return "64a1b2c3d4e5f6789a0b1c2d"  # Valid 24-character ObjectId
    else:
        # For demo/testing purposes, return a default user ID as valid ObjectId
        return "64a1b2c3d4e5f6789a0b1c2d"  # Valid 24-character ObjectId

def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    if not file.filename:
        return False
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    return file_ext in ALLOWED_EXTENSIONS

async def save_file_to_mongodb(file: UploadFile, user_id: str) -> str:
    """Save uploaded file to MongoDB and return file ID"""
    try:
        # Read file content
        file_content = await file.read()
        file.file.seek(0)  # Reset file pointer
        
        # Encode content as base64
        content_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{user_id}_{uuid.uuid4()}{file_ext}"
        
        # Save to MongoDB using Prisma
        db = await get_prisma()
        file_record = await db.filestorage.create(
            data={
                "filename": unique_filename,
                "originalName": file.filename,
                "contentType": file.content_type or "application/octet-stream",
                "fileSize": len(file_content),
                "content": content_base64,
                "userId": user_id
            }
        )
        
        return file_record.id
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

async def get_file_from_mongodb(file_id: str) -> tuple[bytes, str]:
    """Retrieve file content from MongoDB"""
    try:
        db = await get_prisma()
        file_record = await db.filestorage.find_unique(where={"id": file_id})
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Decode base64 content
        file_content = base64.b64decode(file_record.content)
        
        return file_content, file_record.originalName
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")

async def process_csv_file(file_content: bytes) -> tuple[List[dict], int]:
    """Process CSV file content and return customer data"""
    # Create a file-like object from bytes
    csv_buffer = io.StringIO(file_content.decode('utf-8'))
    df = pd.read_csv(csv_buffer)
    customers = df.to_dict(orient='records')
    
    # Transform CSV data to standard format
    processed_customers = []
    for customer in customers:
        processed_customer = {
            "customer_id": customer.get("Contact ID"),
            "name": f"{customer.get('First Name', '')} {customer.get('Last Name', '')}".strip(),
            "email": customer.get("Email"),
            "phone": customer.get("Phone"),
            "address": customer.get("Primary Address"),
            "type": customer.get("Type"),
            "profile": customer.get("profile"),
            "preferences": {
                "property_type": customer.get("property_type"),
                "location": customer.get("location", customer.get("Primary Address")),
                "min_price": customer.get("min_price", 0),
                "max_price": customer.get("max_price", 0),
                "bedrooms": customer.get("bedrooms", 0),
                "bathrooms": customer.get("bathrooms", 0)
            },
            "metadata": {
                "source": "csv_upload",
                "original_data": customer
            }
        }
        processed_customers.append(processed_customer)
    
    return processed_customers, len(processed_customers)

async def process_json_file(file_content: bytes) -> tuple[List[dict], int]:
    """Process JSON file content and return customer data"""
    json_data = json.loads(file_content.decode('utf-8'))
    customers = json_data if isinstance(json_data, list) else [json_data]
    
    # Transform JSON data to standard format
    processed_customers = []
    for customer in customers:
        processed_customer = {
            "customer_id": customer.get("id"),
            "name": customer.get("name"),
            "email": customer.get("email"),
            "phone": customer.get("phone"),
            "address": customer.get("address"),
            "type": customer.get("type"),
            "profile": customer.get("profile"),
            "preferences": customer.get("preferences", {}),
            "metadata": {
                "source": "json_upload",
                "original_data": customer
            }
        }
        processed_customers.append(processed_customer)
    
    return processed_customers, len(processed_customers)

@router.post("/customer-data", response_model=UploadResponse)
async def upload_customer_data(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    sync_to_vector_db: bool = Form(True),
    user_id: str = Depends(get_user_id_from_auth0)
):
    """
    Upload customer data file (CSV or JSON) and optionally sync to vector database
    """
    try:
        # Validate file
        if not validate_file(file):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file format. Only CSV and JSON files are allowed."
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Save file to MongoDB
        file_id = await save_file_to_mongodb(file, user_id)
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Get file content for processing
        file_content, _ = await get_file_from_mongodb(file_id)
        
        # Process file based on type
        if file_ext == ".csv":
            customers_data, record_count = await process_csv_file(file_content)
        elif file_ext == ".json":
            customers_data, record_count = await process_json_file(file_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create data source record in database
        db = await get_prisma()
        data_source = await db.customerdatasource.create(
            data={
                "name": name,
                "description": description,
                "fileId": file_id,
                "fileType": file_ext.lstrip('.'),
                "recordCount": record_count,
                "userId": user_id
            }
        )
        
        # Optionally sync to vector database
        if sync_to_vector_db:
            try:
                # Create a temporary file for the RAG agent
                import tempfile
                with tempfile.NamedTemporaryFile(mode='wb', suffix=file_ext, delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                rag_agent = RealEstateRAGAgent()
                if file_ext == ".csv":
                    rag_agent.load_customers_from_csv(temp_file_path, user_id)
                else:
                    rag_agent.load_customers_from_json(temp_file_path, user_id)
                
                # Clean up temporary file
                os.unlink(temp_file_path)
                
                # Update last sync time
                await db.customerdatasource.update(
                    where={"id": data_source.id},
                    data={"lastSync": datetime.now()}
                )
                
            except Exception as e:
                # Log the error but don't fail the upload
                print(f"Warning: Failed to sync to vector DB: {str(e)}")
        
        return UploadResponse(
            success=True,
            message=f"Successfully uploaded {record_count} customer records",
            file_id=file_id,
            data_source_id=data_source.id,
            record_count=record_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/data-sources", response_model=List[DataSourceInfo])
async def get_data_sources(user_id: str = Depends(get_user_id_from_auth0)):
    """Get all data sources for the authenticated user"""
    try:
        db = await get_prisma()
        data_sources = await db.customerdatasource.find_many(
            where={
                "userId": user_id,
                "isActive": True
            },
            order_by={"createdAt": "desc"}
        )
        
        return [
            DataSourceInfo(
                id=ds.id,
                name=ds.name,
                description=ds.description,
                file_type=ds.fileType,
                record_count=ds.recordCount,
                created_at=ds.createdAt.isoformat(),
                last_sync=ds.lastSync.isoformat() if ds.lastSync else None
            )
            for ds in data_sources
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data sources: {str(e)}")

@router.post("/sync-to-vector/{data_source_id}")
async def sync_to_vector_db(
    data_source_id: str,
    user_id: str = Depends(get_user_id_from_auth0)
):
    """Sync a data source to the vector database"""
    try:
        db = await get_prisma()
        
        # Get data source
        data_source = await db.customerdatasource.find_unique(
            where={"id": data_source_id}
        )
        
        if not data_source or data_source.userId != user_id:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Get file content
        file_content, original_name = await get_file_from_mongodb(data_source.fileId)
        
        # Create temporary file for RAG agent
        import tempfile
        file_ext = f".{data_source.fileType}"
        with tempfile.NamedTemporaryFile(mode='wb', suffix=file_ext, delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Sync to vector database
            rag_agent = RealEstateRAGAgent()
            if data_source.fileType == "csv":
                rag_agent.load_customers_from_csv(temp_file_path, user_id)
            elif data_source.fileType == "json":
                rag_agent.load_customers_from_json(temp_file_path, user_id)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            
            # Update last sync time
            await db.customerdatasource.update(
                where={"id": data_source_id},
                data={"lastSync": datetime.now()}
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
        return {
            "success": True,
            "message": f"Successfully synced {data_source.recordCount} records to vector database",
            "data_source_id": data_source_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@router.delete("/data-source/{data_source_id}")
async def delete_data_source(
    data_source_id: str,
    user_id: str = Depends(get_user_id_from_auth0)
):
    """Delete a data source and its associated file"""
    try:
        db = await get_prisma()
        
        # Get data source
        data_source = await db.customerdatasource.find_unique(
            where={"id": data_source_id}
        )
        
        if not data_source or data_source.userId != user_id:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Delete the file from MongoDB
        if data_source.fileId:
            await db.filestorage.delete(where={"id": data_source.fileId})
        
        # Delete the data source record
        await db.customerdatasource.delete(where={"id": data_source_id})
        
        return {
            "success": True,
            "message": "Data source and associated file deleted successfully",
            "data_source_id": data_source_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@router.get("/template/csv")
async def download_csv_template():
    """Download a CSV template for customer data"""
    template_data = {
        "Contact ID": ["CUST001", "CUST002"],
        "First Name": ["John", "Jane"],
        "Last Name": ["Doe", "Smith"],
        "Email": ["john.doe@email.com", "jane.smith@email.com"],
        "Phone": ["+1-555-0101", "+1-555-0102"],
        "Primary Address": ["123 Main St, New York, NY", "456 Oak Ave, Boston, MA"],
        "Type": ["Buyer", "Seller"]
    }
    
    df = pd.DataFrame(template_data)
    csv_content = df.to_csv(index=False)
    
    return JSONResponse(
        content={"csv_content": csv_content},
        headers={"Content-Disposition": "attachment; filename=customer_template.csv"}
    )

@router.get("/template/json")
async def download_json_template():
    """Download a JSON template for customer data"""
    template_data = [
        {
            "id": "cust_001",
            "name": "John Doe",
            "email": "john.doe@email.com",
            "phone": "+1-555-0101",
            "profile": "Young professional looking for downtown living",
            "preferences": {
                "property_type": "apartment",
                "location": "downtown",
                "price_range": {"min": 800000, "max": 1200000},
                "bedrooms": 2,
                "bathrooms": 2,
                "amenities": ["gym", "pool", "parking"],
                "description": "Looking for modern apartment with city views"
            }
        },
        {
            "id": "cust_002", 
            "name": "Jane Smith",
            "email": "jane.smith@email.com",
            "phone": "+1-555-0102",
            "profile": "Family with children seeking suburban home",
            "preferences": {
                "property_type": "house",
                "location": "suburbs",
                "price_range": {"min": 600000, "max": 900000},
                "bedrooms": 3,
                "bathrooms": 2,
                "amenities": ["garden", "garage", "good schools"],
                "description": "Family home with yard and good school district"
            }
        }
    ]
    
    return JSONResponse(
        content=template_data,
        headers={"Content-Disposition": "attachment; filename=customer_template.json"}
    )
