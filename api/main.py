"""
FastAPI API for Deed Validator
"""
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from propy import (
    DeedValidator, 
    ValidationError, 
    DateLogicError, 
    MoneyMismatchError, 
    CountyNotFoundError
)

# Initialize FastAPI app
app = FastAPI(
    title="Deed Validator API",
    description="API for validating deed documents extracted from OCR text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize validator (singleton)
validator = None

def get_validator() -> DeedValidator:
    """Get or create validator instance"""
    global validator
    if validator is None:
        validator = DeedValidator()
    return validator


# Request/Response Models
class ValidateDeedRequest(BaseModel):
    """Request model for deed validation"""
    ocr_text: str = Field(..., description="Raw OCR text from the deed document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ocr_text": """*** RECORDING REQ ***
Doc: DEED-TRUST-0042
County: LA  |  State: CA
Date Signed: 2024-01-15
Date Recorded: 2024-01-20
Grantor:  John Doe
Grantee:  Jane Doe
Amount: $1,250,000.00 (One Million Two Fifty Thousand Dollars)
APN: 992-001-XA
Status: PRELIMINARY
*** END ***"""
            }
        }


class MatchCountyRequest(BaseModel):
    """Request model for county matching"""
    county_name: str = Field(..., description="County name to match")
    state: Optional[str] = Field(None, description="Optional state abbreviation for disambiguation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "county_name": "S. Clara",
                "state": "CA"
            }
        }


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Deed Validator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /validate": "Validate a deed from OCR text (JSON format, use \\n for newlines)",
            "POST /validate/text": "Validate a deed from OCR text (Form-data format, accepts literal line breaks)",
            "POST /validate/file": "Validate a deed from OCR text file upload",
            "POST /match-county": "Match a county name to canonical name",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        val = get_validator()
        return {
            "status": "healthy",
            "validator_initialized": True,
            "counties_loaded": len(val.counties) > 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/validate", response_model=Dict[str, Any])
async def validate_deed(request: ValidateDeedRequest):
    """
    Validate a deed document from OCR text (JSON format).
    
    Accepts OCR text as a JSON string. Multi-line strings should use \\n for newlines.
    
    Example:
    ```json
    {
      "ocr_text": "*** RECORDING REQ ***\\nDoc: DEED-001\\nCounty: LA\\n..."
    }
    ```
    
    This endpoint:
    1. Extracts structured data from OCR text using LLM
    2. Matches county name to canonical name
    3. Validates date logic, money consistency, etc.
    4. Calculates closing costs
    5. Returns validation result
    """
    try:
        val = get_validator()
        
        # Step 1: Extract structured data from OCR
        extracted = val.extract_with_llm(request.ocr_text)
        
        # Step 2: Validate and enrich
        result = val.validate(extracted)
        
        return result
        
    except DateLogicError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "DateLogicError",
                "message": str(e),
                "type": "date_validation_failed"
            }
        )
    except MoneyMismatchError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "MoneyMismatchError",
                "message": str(e),
                "type": "money_validation_failed"
            }
        )
    except CountyNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "CountyNotFoundError",
                "message": str(e),
                "type": "county_not_found"
            }
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ValidationError",
                "message": str(e),
                "type": "validation_failed"
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ValueError",
                "message": str(e),
                "type": "invalid_input"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"An unexpected error occurred: {str(e)}",
                "type": "server_error"
            }
        )


@app.post("/validate/text", response_model=Dict[str, Any])
async def validate_deed_text(ocr_text: str = Form(..., description="Raw OCR text with line breaks")):
    """
    Validate a deed document from OCR text (Form-data format).
    
    This endpoint accepts OCR text as form-data, allowing you to send multi-line text directly
    without needing to escape newlines as \\n.
    
    Example curl:
    ```bash
    curl -X POST "http://127.0.0.1:8000/validate/text" \\
      -F "ocr_text=*** RECORDING REQ ***
    Doc: DEED-TRUST-0042
    County: S. Clara  |  State: CA
    Date Signed: 2024-01-15
    Date Recorded: 2024-01-20
    ..."
    ```
    """
    try:
        val = get_validator()
        extracted = val.extract_with_llm(ocr_text)
        result = val.validate(extracted)
        return result
    except DateLogicError as e:
        raise HTTPException(status_code=400, detail={"error": "DateLogicError", "message": str(e), "type": "date_validation_failed"})
    except MoneyMismatchError as e:
        raise HTTPException(status_code=400, detail={"error": "MoneyMismatchError", "message": str(e), "type": "money_validation_failed"})
    except CountyNotFoundError as e:
        raise HTTPException(status_code=404, detail={"error": "CountyNotFoundError", "message": str(e), "type": "county_not_found"})
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"error": "ValidationError", "message": str(e), "type": "validation_failed"})
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "ValueError", "message": str(e), "type": "invalid_input"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "InternalServerError", "message": f"An unexpected error occurred: {str(e)}", "type": "server_error"})


@app.post("/validate/file", response_model=Dict[str, Any])
async def validate_deed_file(file: UploadFile = File(..., description="Text file containing OCR text")):
    """
    Validate a deed document from OCR text file upload.
    
    Upload a text file containing the OCR text. This is useful for large documents
    or when you want to preserve exact formatting.
    """
    try:
        content = await file.read()
        ocr_text = content.decode('utf-8')
        val = get_validator()
        extracted = val.extract_with_llm(ocr_text)
        result = val.validate(extracted)
        return result
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail={"error": "UnicodeDecodeError", "message": "File must be UTF-8 encoded text", "type": "invalid_file_encoding"})
    except DateLogicError as e:
        raise HTTPException(status_code=400, detail={"error": "DateLogicError", "message": str(e), "type": "date_validation_failed"})
    except MoneyMismatchError as e:
        raise HTTPException(status_code=400, detail={"error": "MoneyMismatchError", "message": str(e), "type": "money_validation_failed"})
    except CountyNotFoundError as e:
        raise HTTPException(status_code=404, detail={"error": "CountyNotFoundError", "message": str(e), "type": "county_not_found"})
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"error": "ValidationError", "message": str(e), "type": "validation_failed"})
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "ValueError", "message": str(e), "type": "invalid_input"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "InternalServerError", "message": f"An unexpected error occurred: {str(e)}", "type": "server_error"})


@app.post("/match-county")
async def match_county(request: MatchCountyRequest):
    """
    Match a county name to its canonical name and get tax rate.
    
    Uses fuzzy matching with rapidfuzz and falls back to LLM if needed.
    """
    try:
        val = get_validator()
        canonical_name, tax_rate = val.match_county(request.county_name, request.state)
        
        return {
            "input": request.county_name,
            "canonical": canonical_name,
            "tax_rate": tax_rate,
            "state": request.state
        }
        
    except CountyNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "CountyNotFoundError",
                "message": str(e),
                "input": request.county_name,
                "state": request.state
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ValueError",
                "message": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"An unexpected error occurred: {str(e)}"
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
