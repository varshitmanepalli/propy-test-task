# Propy Deed Validator

A professional solution for validating OCR-scanned property deeds using LLM extraction followed by logic validation.

## Overview

This project addresses the critical need to validate AI-extracted data from OCR-scanned property deeds before recording transactions. The solution combines the flexibility of LLM-based extraction with code-based validation to catch errors that could lead to fraudulent transactions.

## Architecture

The solution follows a three-stage pipeline:

1. **Extraction**: Use an LLM (OpenAI GPT-4o-mini) to parse messy OCR text into structured data
2. **Enrichment**: Match abbreviated county names to canonical names and retrieve tax rates
3. **Validation**: Run code-based sanity checks to catch logical errors

## Project Structure

```
propy-test-task/
├── src/
│   └── propy/
│       ├── __init__.py          # Package initialization and exports
│       ├── exceptions.py        # Custom exception classes
│       ├── models.py            # Data models (ExtractedDeed)
│       └── validator.py         # Main validator implementation
├── api/
│   └── main.py                  # FastAPI REST API
├── config/
│   ├── counties.json            # County tax rate reference data
│   ├── validation_rules.json   # Declarative validation rules
│   └── prompts.yaml            # LLM prompts configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Key Design Decisions

### 1. LLM for Extraction, Code for Validation

**Why**: LLMs excel at parsing unstructured text and handling variations, but we cannot trust them for critical financial logic. All validation is done in deterministic Python code.

- **Extraction**: LLM handles messy OCR text, abbreviations, and formatting variations
- **Validation**: Code-based checks ensure correctness (dates, amounts, logic)

### 2. County Name Matching

**Problem**: OCR text says "S. Clara" but our database has "Santa Clara"

**Solution**: Implemented sophisticated fuzzy matching algorithm that:
- Uses multiple rapidfuzz strategies (partial_ratio, ratio, token_sort_ratio)
- Validates word overlap to prevent false matches
- Falls back to LLM only when confidence is low
- Handles abbreviations like "S. Clara" → "Santa Clara" without relying solely on AI

**Note**: The `config/counties.json` file includes counties from the state of California.

### 3. Rule-Based Validation System

**Problem**: Hardcoded validation logic is difficult to maintain and extend

**Solution**: Declarative JSON configuration (`config/validation_rules.json`) that supports:
- **Comparison Rules**: Field-to-field comparisons with operators (e.g., `date_recorded < date_signed`)
- **Custom Rules**: Delegation to specific validation functions (e.g., `validate_money`)
- **Enrichment Rules**: Calculated fields based on formulas (e.g., `closing_costs`)

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd propy-test-task
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Or set the environment variable directly:
```bash
export OPENAI_API_KEY="your-api-key-here"  # On Windows: set OPENAI_API_KEY=your-api-key-here
```

## Usage

### As a Python Package

```python
import sys
from pathlib import Path

# Add src to path if running as script
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from propy import DeedValidator, ExtractedDeed

# Initialize validator (automatically uses config files from config/ directory)
validator = DeedValidator()

# Extract and validate
ocr_text = """
*** RECORDING REQ ***
Doc: DEED-TRUST-0042
County: S. Clara  |  State: CA
Date Signed: 2024-01-15
Date Recorded: 2024-01-20
Grantor:  John Doe
Grantee:  Jane Doe
Amount: $1,250,000.00 (One Million Two Fifty Thousand Dollars)
APN: 992-001-XA
Status: PRELIMINARY
*** END ***
"""

# Extract structured data
extracted = validator.extract_with_llm(ocr_text)

# Validate and enrich
result = validator.validate(extracted)
print(result)
```

### As a REST API

1. **Start the API server**:

From the project root:
```bash
python -m api.main
```

Or using uvicorn directly:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Access the API**:
- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc

3. **Example API calls**:

**Validate a deed (JSON)**:
```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{"ocr_text": "*** RECORDING REQ ***\\nDoc: DEED-001\\nCounty: LA\\n..."}'
```

**Validate a deed (Form-data with line breaks)**:
```bash
curl -X POST "http://localhost:8000/validate/text" \
  -F "ocr_text=*** RECORDING REQ ***
Doc: DEED-TRUST-0042
County: S. Clara  |  State: CA
..."
```

## Configuration

### Validation Rules (`config/validation_rules.json`)

Define validation and enrichment rules declaratively:

```json
{
  "validation_rules": [
    {
      "name": "date_logic",
      "type": "comparison",
      "field1": "date_recorded",
      "field2": "date_signed",
      "operator": "<",
      "error_type": "DateLogicError",
      "error_message": "Invalid date logic: Document was recorded on {date_recorded} but signed on {date_signed}."
    }
  ],
  "enrichment_rules": [
    {
      "name": "closing_costs",
      "type": "calculation",
      "formula": "{amount_numeric} * {tax_rate}",
      "output_field": "closing_costs"
    }
  ]
}
```

### Prompts (`config/prompts.yaml`)

Manage all LLM prompts in one place:

```yaml
prompts:
  extraction:
    system: "You are a precise data extraction assistant..."
    user_template: |
      Extract structured data from OCR text...
      {ocr_text}
```

## API Endpoints

### POST `/validate`
Validate a deed from OCR text (JSON format)

### POST `/validate/text`
Validate a deed from OCR text (Form-data format, accepts literal line breaks)

### POST `/validate/file`
Validate a deed from uploaded text file

### POST `/match-county`
Match a county name to canonical name and get tax rate

### GET `/health`
Health check endpoint

### GET `/`
API information and available endpoints

## Error Handling

The validator raises specific exceptions for different error types:

- `ValidationError`: Base exception for all validation errors
- `DateLogicError`: Raised when recorded date is before signed date
- `MoneyMismatchError`: Raised when written amount doesn't match numeric amount
- `CountyNotFoundError`: Raised when county cannot be matched

## Why This Approach?

1. **Paranoid Engineering**: We don't trust the LLM for critical logic - we use it for what it's good at (parsing) and validate with code
2. **Transparency**: All validation logic is explicit and auditable
3. **Maintainability**: Easy to add new validation rules without touching LLM prompts
4. **Reliability**: Deterministic code will catch errors consistently, regardless of LLM behavior
5. **Professional Structure**: Clean architecture that scales and is easy to understand
Contributions are welcome! Please feel free to submit a Pull Request.
