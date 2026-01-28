"""
The "Bad Deed" Validator

A paranoid engineering approach to validating OCR-scanned deeds using LLM extraction
followed by rigorous code-based validation.
"""

import json
import re
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI library not installed. Install with: pip install openai")
    OpenAI = None


class ValidationError(Exception):
    """Base exception for validation errors"""
    pass


class DateLogicError(ValidationError):
    """Raised when recorded date is before signed date"""
    pass


class MoneyMismatchError(ValidationError):
    """Raised when written amount doesn't match numeric amount"""
    pass


class CountyNotFoundError(ValidationError):
    """Raised when county cannot be matched"""
    pass


@dataclass
class ExtractedDeed:
    """Structured representation of a deed"""
    doc_number: str
    county: str
    state: str
    date_signed: datetime
    date_recorded: datetime
    grantor: str
    grantee: str
    amount_numeric: float
    amount_written: str
    apn: str
    status: str


class DeedValidator:
    """Validates deeds extracted from OCR text"""
    
    def __init__(self, counties_file: str = "counties.json", api_key: Optional[str] = None):
        """
        Initialize the validator.
        
        Args:
            counties_file: Path to counties JSON file
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.counties = self._load_counties(counties_file)
        self.client = None
        if OpenAI:
            self.client = OpenAI(api_key=api_key)
    
    def _load_counties(self, counties_file: str) -> Dict[str, float]:
        """Load counties data and create a lookup dictionary"""
        with open(counties_file, 'r') as f:
            counties_list = json.load(f)
        return {county['name']: county['tax_rate'] for county in counties_list}
    
    def extract_with_llm(self, ocr_text: str) -> ExtractedDeed:
        """
        Use LLM to extract structured data from messy OCR text.
        
        Args:
            ocr_text: Raw OCR text
            
        Returns:
            ExtractedDeed object
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY or pass api_key.")
        
        prompt = f"""Extract the following information from this OCR-scanned deed text and return it as a JSON object:

{ocr_text}

Extract:
- doc_number: Document number (e.g., "DEED-TRUST-0042")
- county: County name exactly as written
- state: State abbreviation (e.g., "CA")
- date_signed: Date signed in YYYY-MM-DD format
- date_recorded: Date recorded in YYYY-MM-DD format
- grantor: Grantor name
- grantee: Grantee name
- amount_numeric: Numeric amount as a number (no currency symbols)
- amount_written: Written amount exactly as it appears in the text
- apn: APN number
- status: Status

Return ONLY valid JSON, no markdown, no explanations."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise data extraction assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        json_str = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        json_str = re.sub(r'^```json\n?', '', json_str)
        json_str = re.sub(r'^```\n?', '', json_str)
        json_str = re.sub(r'\n?```$', '', json_str)
        
        data = json.loads(json_str)
        
        return ExtractedDeed(
            doc_number=data['doc_number'],
            county=data['county'],
            state=data['state'],
            date_signed=datetime.strptime(data['date_signed'], '%Y-%m-%d'),
            date_recorded=datetime.strptime(data['date_recorded'], '%Y-%m-%d'),
            grantor=data['grantor'],
            grantee=data['grantee'],
            amount_numeric=float(data['amount_numeric']),
            amount_written=data['amount_written'],
            apn=data['apn'],
            status=data['status']
        )
    
    def match_county(self, ocr_county: str) -> Tuple[str, float]:
        """
        Match OCR county name to canonical county name using fuzzy matching.
        
        Handles abbreviations like "S. Clara" -> "Santa Clara"
        
        Args:
            ocr_county: County name from OCR
            
        Returns:
            Tuple of (canonical_county_name, tax_rate)
            
        Raises:
            CountyNotFoundError: If county cannot be matched
        """
        ocr_county = ocr_county.strip()
        
        # Direct match first
        if ocr_county in self.counties:
            return ocr_county, self.counties[ocr_county]
        
        # Normalize for fuzzy matching
        ocr_normalized = ocr_county.lower().replace('.', '').replace(' ', '')
        
        for canonical_name in self.counties.keys():
            canonical_normalized = canonical_name.lower().replace(' ', '')
            
            # Check if OCR county is an abbreviation of canonical name
            # e.g., "S. Clara" -> "Santa Clara"
            if canonical_normalized.startswith(ocr_normalized) or ocr_normalized in canonical_normalized:
                # Additional check: ensure it's a reasonable match
                # Split by common delimiters and check if parts match
                ocr_parts = re.split(r'[.\s]+', ocr_county.lower())
                canonical_parts = canonical_name.lower().split()
                
                # Check if all OCR parts appear in canonical name
                if all(any(ocr_part in canon_part for canon_part in canonical_parts) 
                       for ocr_part in ocr_parts if ocr_part):
                    return canonical_name, self.counties[canonical_name]
        
        raise CountyNotFoundError(f"Could not match county '{ocr_county}' to any known county")
    
    def validate_date_logic(self, date_signed: datetime, date_recorded: datetime):
        """
        Validate that date_recorded is not before date_signed.
        
        Args:
            date_signed: Date the document was signed
            date_recorded: Date the document was recorded
            
        Raises:
            DateLogicError: If recorded date is before signed date
        """
        if date_recorded < date_signed:
            raise DateLogicError(
                f"Invalid date logic: Document was recorded on {date_recorded.date()} "
                f"but signed on {date_signed.date()}. Recording cannot occur before signing."
            )
    
    def words_to_number(self, words: str) -> float:
        """
        Convert written number words to numeric value.
        
        Args:
            words: Written number (e.g., "One Million Two Hundred Thousand Dollars")
            
        Returns:
            Numeric value as float
        """
        # Remove common suffixes and normalize
        words = words.lower().replace('dollars', '').replace('dollar', '').strip()
        
        # Number word mappings
        word_map = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
            'hundred': 100, 'thousand': 1000, 'million': 1000000,
            'billion': 1000000000
        }
        
        # Split into tokens
        tokens = words.replace('-', ' ').split()
        
        result = 0
        current = 0
        
        for token in tokens:
            if token in word_map:
                value = word_map[token]
                if value == 100:
                    current *= 100
                elif value >= 1000:
                    result += current * value
                    current = 0
                else:
                    current += value
            else:
                # Try to handle compound numbers like "twenty-five"
                parts = token.split('-')
                if len(parts) == 2:
                    if parts[0] in word_map and parts[1] in word_map:
                        current += word_map[parts[0]] + word_map[parts[1]]
        
        result += current
        return float(result)
    
    def validate_money(self, numeric_amount: float, written_amount: str, tolerance: float = 0.01):
        """
        Validate that written amount matches numeric amount.
        
        Args:
            numeric_amount: Numeric amount from document
            written_amount: Written amount in words
            tolerance: Allowed difference (default $0.01)
            
        Raises:
            MoneyMismatchError: If amounts don't match within tolerance
        """
        written_numeric = self.words_to_number(written_amount)
        difference = abs(numeric_amount - written_numeric)
        
        if difference > tolerance:
            raise MoneyMismatchError(
                f"Amount mismatch: Numeric amount is ${numeric_amount:,.2f} "
                f"but written amount translates to ${written_numeric:,.2f}. "
                f"Difference: ${difference:,.2f}"
            )
    
    def validate(self, extracted_deed: ExtractedDeed) -> Dict:
        """
        Run all validation checks on extracted deed.
        
        Args:
            extracted_deed: Extracted deed data
            
        Returns:
            Dictionary with validation results and enriched data
            
        Raises:
            ValidationError: If any validation fails
        """
        # Enrich county data
        canonical_county, tax_rate = self.match_county(extracted_deed.county)
        
        # Validate date logic (code-based, not AI)
        self.validate_date_logic(extracted_deed.date_signed, extracted_deed.date_recorded)
        
        # Validate money consistency (code-based, not AI)
        self.validate_money(extracted_deed.amount_numeric, extracted_deed.amount_written)
        
        # Calculate closing costs (example enrichment)
        closing_costs = extracted_deed.amount_numeric * tax_rate
        
        return {
            'valid': True,
            'doc_number': extracted_deed.doc_number,
            'county': {
                'ocr': extracted_deed.county,
                'canonical': canonical_county,
                'tax_rate': tax_rate
            },
            'state': extracted_deed.state,
            'date_signed': extracted_deed.date_signed.isoformat(),
            'date_recorded': extracted_deed.date_recorded.isoformat(),
            'grantor': extracted_deed.grantor,
            'grantee': extracted_deed.grantee,
            'amount': {
                'numeric': extracted_deed.amount_numeric,
                'written': extracted_deed.amount_written,
                'validated': True
            },
            'apn': extracted_deed.apn,
            'status': extracted_deed.status,
            'closing_costs': closing_costs
        }


def main():
    """Main entry point"""
    import os
    
    # The exact OCR text from the task
    OCR_TEXT = """*** RECORDING REQ ***
Doc: DEED-TRUST-0042
County: S. Clara | State: CA
Date Signed: 2024-01-15
Date Recorded: 2024-01-10
Grantor: T.E.S.L.A. Holdings LLC
Grantee: John & Sarah Connor
Amount: $1,250,000.00 (One Million Two Hundred Thousand Dollars)
APN: 992-001-XA
Status: PRELIMINARY
*** END ***"""
    
    # Get API key from environment or prompt
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it as an environment variable.")
        print("You can also pass it directly to DeedValidator(api_key='...')")
        return
    
    validator = DeedValidator(api_key=api_key)
    
    print("=" * 60)
    print("Deed Validator - Processing OCR Text")
    print("=" * 60)
    print("\nRaw OCR Text:")
    print(OCR_TEXT)
    print("\n" + "=" * 60)
    
    try:
        # Step 1: Extract with LLM
        print("\n[Step 1] Extracting data with LLM...")
        extracted = validator.extract_with_llm(OCR_TEXT)
        print("✓ Extraction complete")
        
        # Step 2 & 3: Validate (includes enrichment and sanity checks)
        print("\n[Step 2-3] Validating and enriching data...")
        result = validator.validate(extracted)
        print("✓ Validation complete")
        
        print("\n" + "=" * 60)
        print("VALIDATION RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        
    except DateLogicError as e:
        print(f"\n❌ DATE LOGIC ERROR: {e}")
    except MoneyMismatchError as e:
        print(f"\n❌ MONEY MISMATCH ERROR: {e}")
    except CountyNotFoundError as e:
        print(f"\n❌ COUNTY NOT FOUND ERROR: {e}")
    except ValidationError as e:
        print(f"\n❌ VALIDATION ERROR: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
