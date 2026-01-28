# The "Bad Deed" Validator

A paranoid engineering solution for validating OCR-scanned deeds using LLM extraction followed by rigorous code-based validation.

## Overview

This project addresses the critical need to validate AI-extracted data from OCR-scanned property deeds before recording transactions on the blockchain. The solution combines the flexibility of LLM-based extraction with deterministic code-based validation to catch errors that could lead to fraudulent transactions.

## Architecture

The solution follows a three-stage pipeline:

1. **Extraction**: Use an LLM (OpenAI GPT-4o-mini) to parse messy OCR text into structured data
2. **Enrichment**: Match abbreviated county names to canonical names and retrieve tax rates
3. **Validation**: Run deterministic code-based sanity checks to catch logical errors

## Key Design Decisions

### 1. LLM for Extraction, Code for Validation

**Why**: LLMs excel at parsing unstructured text and handling variations, but we cannot trust them for critical financial logic. All validation is done in deterministic Python code.

- **Extraction**: LLM handles messy OCR text, abbreviations, and formatting variations
- **Validation**: Code-based checks ensure correctness (dates, amounts, logic)

### 2. County Name Matching

**Problem**: OCR text says "S. Clara" but our database has "Santa Clara"

**Solution**: Implemented fuzzy matching algorithm that:
- First tries direct match
- Then normalizes both strings (removes punctuation, spaces)
- Checks if OCR county is an abbreviation of canonical name
- Validates that all parts of the OCR name appear in the canonical name

This handles abbreviations like "S. Clara" → "Santa Clara" without relying on AI interpretation.

### 3. Date Logic Validation

**Problem**: Document recorded (Jan 10) before it was signed (Jan 15) - impossible!

**Solution**: Pure code-based date comparison:
```python
if date_recorded < date_signed:
    raise DateLogicError(...)
```

This is **not** delegated to the LLM - it's deterministic Python logic that will always catch this error.

### 4. Money Validation

**Problem**: Digits say $1,250,000 but words say "One Million Two Hundred Thousand" ($1,200,000) - $50k discrepancy

**Solution**: Implemented a `words_to_number()` function that:
- Parses written number words deterministically
- Converts to numeric value
- Compares with the numeric amount from the document
- Raises `MoneyMismatchError` if difference exceeds tolerance ($0.01)

This ensures we **flag** discrepancies rather than silently choosing one value.

## File Structure

```
propy-test-task/
├── deed_validator.py    # Main validation script
├── counties.json        # County tax rate reference data
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or pass it directly when instantiating `DeedValidator(api_key='...')`

## Usage

Run the validator:
```bash
python deed_validator.py
```

The script will:
1. Extract data from the OCR text using the LLM
2. Match "S. Clara" to "Santa Clara" and retrieve tax rate
3. Validate date logic (will catch the Jan 10 < Jan 15 error)
4. Validate money consistency (will catch the $50k discrepancy)

## Expected Output

The script will catch both validation errors:

1. **DateLogicError**: "Invalid date logic: Document was recorded on 2024-01-10 but signed on 2024-01-15"
2. **MoneyMismatchError**: "Amount mismatch: Numeric amount is $1,250,000.00 but written amount translates to $1,200,000.00. Difference: $50,000.00"

## Engineering Hygiene Highlights

### Error Handling
- Custom exception hierarchy (`ValidationError` → `DateLogicError`, `MoneyMismatchError`, `CountyNotFoundError`)
- Clear, actionable error messages
- Proper exception propagation

### Code Structure
- Separation of concerns: extraction, enrichment, validation
- Dataclass for structured data (`ExtractedDeed`)
- Type hints throughout
- Well-documented functions

### Deterministic Validation
- Date validation: Pure Python datetime comparison
- Money validation: Deterministic word-to-number conversion
- County matching: Rule-based fuzzy matching (not AI-dependent)

### Testability
- Each validation function can be tested independently
- No hidden state or side effects in validation logic
- Clear input/output contracts

## Why This Approach?

1. **Paranoid Engineering**: We don't trust the LLM for critical logic - we use it for what it's good at (parsing) and validate with code
2. **Transparency**: All validation logic is explicit and auditable
3. **Maintainability**: Easy to add new validation rules without touching LLM prompts
4. **Reliability**: Deterministic code will catch errors consistently, regardless of LLM behavior

## Future Enhancements

- Add unit tests for each validation function
- Support for additional LLM providers (Anthropic, etc.)
- More sophisticated county matching (fuzzy string matching libraries)
- API endpoint wrapper (Flask/FastAPI)
- Batch processing support
- Logging and audit trail
