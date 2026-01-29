"""
Deed validator implementation with LLM extraction and rule-based validation.
"""

import json
import re
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
from operator import lt, le, gt, ge, eq, ne
from decimal import Decimal

try:
    from rapidfuzz import fuzz
except ImportError:
    print("Warning: rapidfuzz library not installed. Install with: pip install rapidfuzz")
    fuzz = None

try:
    import yaml
except ImportError:
    print("Warning: pyyaml library not installed. Install with: pip install pyyaml")
    yaml = None

try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI library not installed. Install with: pip install openai")
    OpenAI = None

# Custom number word conversion constants
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}

_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "quadrillion": 1_000_000_000_000_000,
}

_IGNORED = {
    "and", "of", "the", "a", "an", "dollars", "dollar"
}

import dotenv
dotenv.load_dotenv()


from .exceptions import ValidationError, DateLogicError, MoneyMismatchError, CountyNotFoundError
from .models import ExtractedDeed


class DeedValidator:
    """Validates deeds extracted from OCR text"""
    
    # Operator mapping for comparison rules
    OPERATORS = {
        '<': lt,
        '<=': le,
        '>': gt,
        '>=': ge,
        '==': eq,
        '!=': ne
    }
    
    def __init__(
        self, 
        counties_file: Optional[str] = None, 
        validation_rules_file: Optional[str] = None,
        prompts_file: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the validator.
        
        Args:
            counties_file: Path to counties JSON file (default: config/counties.json)
            validation_rules_file: Path to validation rules JSON file (default: config/validation_rules.json)
            prompts_file: Path to prompts YAML file (default: config/prompts.yaml)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        # Get project root directory (assuming config files are in project root/config/)
        project_root = Path(__file__).parent.parent.parent
        
        # Set default paths relative to project root
        if counties_file is None:
            counties_file = str(project_root / "config" / "counties.json")
        if validation_rules_file is None:
            validation_rules_file = str(project_root / "config" / "validation_rules.json")
        if prompts_file is None:
            prompts_file = str(project_root / "config" / "prompts.yaml")
        
        self.counties = self._load_counties(counties_file)
        self.validation_rules = self._load_validation_rules(validation_rules_file)
        self.prompts = self._load_prompts(prompts_file)
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not set. Set OPENAI_API_KEY or pass api_key.")
        
        if OpenAI is None:
            raise ValueError("OpenAI library not installed. Install with: pip install openai")
        
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def _load_counties(self, counties_file: str) -> Dict[str, float]:
        """Load counties data and create a lookup dictionary"""
        with open(counties_file, 'r') as f:
            counties_list = json.load(f)
        return {county['name']: county['tax_rate'] for county in counties_list}
    
    def _load_validation_rules(self, rules_file: str) -> Dict[str, List[Dict]]:
        """Load validation rules from JSON file"""
        try:
            with open(rules_file, 'r') as f:
                rules_data = json.load(f)
            return rules_data
        except FileNotFoundError:
            # Return default rules if file doesn't exist
            return {
                "validation_rules": [],
                "enrichment_rules": []
            }
    
    def _load_prompts(self, prompts_file: str) -> Dict[str, Dict[str, str]]:
        """Load prompts from YAML file"""
        if yaml is None:
            raise ValueError("pyyaml library is required. Install with: pip install pyyaml")
        
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts_data = yaml.safe_load(f)
            return prompts_data.get('prompts', {})
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts file '{prompts_file}' not found!")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing prompts YAML file: {e}")
    
    def _get_prompt(self, prompt_category: str, prompt_type: str, **kwargs) -> str:
        """
        Get a prompt template and format it with provided variables.
        
        Args:
            prompt_category: Category of prompt (e.g., 'extraction', 'county_matching')
            prompt_type: Type of prompt ('system' or 'user_template')
            **kwargs: Variables to format into the template
            
        Returns:
            Formatted prompt string
        """
        if prompt_category not in self.prompts:
            raise ValueError(f"Prompt category '{prompt_category}' not found in prompts file")
        
        if prompt_type not in self.prompts[prompt_category]:
            raise ValueError(f"Prompt type '{prompt_type}' not found in category '{prompt_category}'")
        
        template = self.prompts[prompt_category][prompt_type]
        
        # Format template with provided variables
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable '{e}' for prompt template")
    
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
        
        # Get prompts from YAML file
        system_prompt = self._get_prompt("extraction", "system")
        user_prompt = self._get_prompt("extraction", "user_template", ocr_text=ocr_text)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
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
    
    def _match_county_with_llm(self, county_name: str, state: Optional[str] = None) -> Tuple[str, float]:
        """
        Use LLM to match county name when fuzzy matching is not confident.
        
        Args:
            county_name: County name from OCR
            state: Optional state abbreviation to help LLM disambiguate
            
        Returns:
            Tuple of (canonical_county_name, tax_rate)
            
        Raises:
            CountyNotFoundError: If county cannot be matched
        """
        if not self.client:
            raise CountyNotFoundError(f"Could not match county '{county_name}' - LLM not available")
        
        # Filter counties by state if state is provided (helps LLM disambiguate)
        # Note: This is a heuristic since counties.json doesn't have state info
        # We'll provide top fuzzy matches as context instead
        top_fuzzy_matches = []
        if fuzz:
            fuzzy_scores = [(name, fuzz.partial_ratio(county_name.lower(), name.lower())) 
                          for name in self.counties.keys()]
            fuzzy_scores.sort(key=lambda x: x[1], reverse=True)
            top_fuzzy_matches = [name for name, score in fuzzy_scores[:5] if score > 50]
        
        state_context = f" in state {state}" if state else ""
        matches_context = f"\nTop fuzzy matches: {', '.join(top_fuzzy_matches)}" if top_fuzzy_matches else ""
        available_counties_sample = ', '.join(sorted(self.counties.keys())[:30])
        total_counties = len(self.counties) - 30
        
        # Get prompts from YAML file
        system_prompt = self._get_prompt("county_matching", "system")
        user_prompt = self._get_prompt(
            "county_matching", 
            "user_template",
            county_name=county_name,
            state_context=state_context,
            available_counties_sample=available_counties_sample,
            total_counties=total_counties,
            matches_context=matches_context
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        llm_county = response.choices[0].message.content.strip()
        # Remove any markdown or extra formatting
        llm_county = re.sub(r'^```\w*\n?', '', llm_county)
        llm_county = re.sub(r'\n?```$', '', llm_county)
        llm_county = llm_county.strip().strip('"').strip("'")
        
        if llm_county.lower() == 'null' or not llm_county:
            raise CountyNotFoundError(f"LLM could not determine county for '{county_name}'")
        
        # Try to match LLM result to our counties (case-insensitive)
        llm_county_lower = llm_county.lower()
        for canonical_name in self.counties.keys():
            if canonical_name.lower() == llm_county_lower:
                return canonical_name, self.counties[canonical_name]
        
        # Try fuzzy match on LLM result as fallback
        if fuzz:
            best_match = None
            best_score = 0
            for canonical_name in self.counties.keys():
                score = fuzz.ratio(llm_county_lower, canonical_name.lower())
                if score > best_score:
                    best_score = score
                    best_match = canonical_name
            
            if best_match and best_score >= 80:
                return best_match, self.counties[best_match]
        
        raise CountyNotFoundError(f"LLM returned '{llm_county}' but it's not in our county database")
    
    def _validate_word_overlap(self, input_name: str, canonical_name: str) -> bool:
        """
        Validate that significant words from input appear in canonical name.
        Helps prevent false matches like "san jose" -> "Santa Clara"
        
        Args:
            input_name: Input county name
            canonical_name: Canonical county name to check against
            
        Returns:
            True if there's good word overlap, False otherwise
        """
        # Split into words and filter out common words/abbreviations
        input_words = [w.strip('.,') for w in input_name.lower().split() if len(w.strip('.,')) > 1]
        canonical_words = [w.strip('.,') for w in canonical_name.lower().split() if len(w.strip('.,')) > 1]
        
        if not input_words:
            return True  # Single character or empty, skip validation
        
        # Check if all significant input words have matches in canonical name
        # Allow partial matches (e.g., "s" matches "santa", "sta" matches "santa")
        matched_words = 0
        for input_word in input_words:
            # Skip very short words unless they're abbreviations
            if len(input_word) <= 1:
                continue
            
            # Check if word appears in any canonical word (allowing partial matches)
            word_matched = False
            for canon_word in canonical_words:
                # Exact match or word is contained in canonical word (for abbreviations)
                if input_word == canon_word or input_word in canon_word:
                    word_matched = True
                    break
                # Or canonical word starts with input word (for abbreviations like "S" -> "Santa")
                if canon_word.startswith(input_word):
                    word_matched = True
                    break
            
            if word_matched:
                matched_words += 1
        
        # Require at least 50% of significant words to match
        if len(input_words) > 0:
            return matched_words / len(input_words) >= 0.5
        return True
    
    def match_county(self, county_name: str, state: Optional[str] = None) -> Tuple[str, float]:
        """
        Match county name to canonical county name using rapidfuzz scoring.
        Falls back to LLM only if confidence is low.
        
        Args:
            county_name: County name from OCR
            state: Optional state abbreviation to help disambiguate
            
        Returns:
            Tuple of (canonical_county_name, tax_rate)
            
        Raises:
            CountyNotFoundError: If county cannot be matched
        """
        if fuzz is None:
            raise ValueError("rapidfuzz library is required. Install with: pip install rapidfuzz")
        
        county_name = county_name.strip()
        
        if not county_name:
            raise CountyNotFoundError("County name cannot be empty")
        
        # Direct match first (fastest path)
        if county_name in self.counties:
            return county_name, self.counties[county_name]
        
        # Case-insensitive direct match
        county_lower = county_name.lower()
        for canonical_name in self.counties.keys():
            if canonical_name.lower() == county_lower:
                return canonical_name, self.counties[canonical_name]
        
        # Use multiple rapidfuzz strategies for better matching
        county_scores = []
        for canonical_name in self.counties.keys():
            input_lower = county_name.lower()
            canon_lower = canonical_name.lower()
            
            # Use multiple scoring methods and take the best
            partial_score = fuzz.partial_ratio(input_lower, canon_lower)
            ratio_score = fuzz.ratio(input_lower, canon_lower)
            token_sort_score = fuzz.token_sort_ratio(input_lower, canon_lower)
            
            # Weighted combination: partial_ratio is best for abbreviations
            # but we also consider full ratio and token_sort for better accuracy
            combined_score = max(
                partial_score,  # Best for abbreviations like "S. Clara"
                ratio_score * 0.9,  # Good for full matches
                token_sort_score * 0.85  # Good for word order variations
            )
            
            county_scores.append((canonical_name, combined_score, partial_score, ratio_score))
        
        # Sort by combined score (highest first)
        county_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not county_scores:
            raise CountyNotFoundError(f"Could not match county '{county_name}' to any known county")
        
        top_match, top_score, top_partial, top_ratio = county_scores[0]
        second_score = county_scores[1][1] if len(county_scores) > 1 else 0
        
        # Confidence threshold: top score >= 85 and clear winner (second score is significantly lower)
        # Also validate word overlap to prevent false matches
        confidence_threshold = 85
        score_gap_threshold = 8  # Increased gap threshold for better accuracy
        
        if top_score >= confidence_threshold and (top_score - second_score) >= score_gap_threshold:
            # Validate word overlap for medium-confidence matches
            if top_score < 95:
                if not self._validate_word_overlap(county_name, top_match):
                    # Word overlap validation failed, use LLM instead
                    return self._match_county_with_llm(county_name, state)
            
            # High confidence match - use fuzzy match result
            return top_match, self.counties[top_match]
        else:
            # Low confidence - use LLM as fallback
            return self._match_county_with_llm(county_name, state)
    
    def _evaluate_comparison_rule(self, rule: Dict, deed_data: Dict) -> Tuple[bool, Optional[str]]:
        """
        Evaluate a comparison rule dynamically.
        
        Args:
            rule: Rule definition with field1, field2, operator
            deed_data: Dictionary of deed field values
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        field1_value = deed_data.get(rule['field1'])
        field2_value = deed_data.get(rule['field2'])
        operator = self.OPERATORS.get(rule['operator'])
        
        if operator is None:
            raise ValueError(f"Unknown operator: {rule['operator']}")
        
        if field1_value is None or field2_value is None:
            return True, None  # Skip validation if fields are missing
        
        is_valid = not operator(field1_value, field2_value)
        
        if not is_valid:
            # Format error message with proper date formatting
            format_data = {}
            for key, value in deed_data.items():
                if isinstance(value, datetime):
                    format_data[key] = value.date()
                else:
                    format_data[key] = value
            error_msg = rule.get('error_message', '').format(**format_data)
            return False, error_msg
        
        return True, None
    
    def _evaluate_custom_rule(self, rule: Dict, deed_data: Dict) -> Tuple[bool, Optional[str]]:
        """
        Evaluate a custom rule by calling the specified function.
        
        Args:
            rule: Rule definition with function name and fields
            deed_data: Dictionary of deed field values
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        function_name = rule.get('function')
        if not function_name:
            return True, None  # Skip if no function specified
        
        # Get the function from this class
        if not hasattr(self, function_name):
            return True, None  # Skip if function doesn't exist
        
        func = getattr(self, function_name)
        
        # Get field values from deed_data
        fields = rule.get('fields', [])
        field_values = []
        for field in fields:
            if field in deed_data:
                field_values.append(deed_data[field])
            else:
                return True, None  # Skip if required field is missing
        
        # Get additional parameters from rule (like tolerance)
        kwargs = {}
        for key, value in rule.items():
            if key not in ['name', 'type', 'function', 'fields', 'error_type', 'error_message']:
                kwargs[key] = value
        
        try:
            # Call the function with field values and kwargs
            func(*field_values, **kwargs)
            return True, None  # Function didn't raise exception, so it's valid
        except ValidationError as e:
            return False, str(e)  # Return the error message
        except Exception as e:
            # Unexpected error - return generic message
            return False, f"Validation error in {function_name}: {str(e)}"
    
    def _parse_integer_words(self, words: str) -> int:
        """
        Parse integer words into a numeric value.
        
        Args:
            words: Space-separated number words
            
        Returns:
            Integer value
        """
        total = 0
        current = 0
        
        # Clean up words - remove any remaining punctuation and normalize
        words = re.sub(r'[^\w\s-]', '', words)
        words = re.sub(r'\s+', ' ', words).strip()
        
        for word in words.split():
            # Remove any remaining punctuation from word
            word = word.strip('.,;:!?()[]{}')
            
            if not word:  # Skip empty strings
                continue
                
            if word in _IGNORED:
                continue
            elif word in _UNITS:
                current += _UNITS[word]
            elif word in _TENS:
                current += _TENS[word]
            elif word == "hundred":
                current *= 100
            elif word in _SCALES:
                total += current * _SCALES[word]
                current = 0
            elif word.isdigit():
                current += int(word)
            else:
                # Try to handle compound numbers like "twenty-five"
                if '-' in word:
                    parts = word.split('-')
                    if len(parts) == 2:
                        part1, part2 = parts[0].strip(), parts[1].strip()
                        if part1 in _TENS and part2 in _UNITS:
                            current += _TENS[part1] + _UNITS[part2]
                            continue
                raise ValueError(f"Unrecognized number word: '{word}'")
        
        return total + current
    
    def words_to_number(self, words: str) -> float:
        """
        Convert written number words to numeric value.
        
        Supports large numbers and cents. Handles numbers like:
        - "One Million Two Hundred Thousand" -> 1,200,000
        - "One Million Two Hundred Fifty Thousand" -> 1,250,000
        - "Two Hundred Fifty Thousand" -> 250,000
        - "One Hundred Dollars and Fifty Cents" -> 100.50
        - "(One million two hundred fifty thousand dollars)" -> 1,250,000
        
        Args:
            words: Written number (e.g., "One Million Two Hundred Thousand Dollars")
            
        Returns:
            Numeric value as float
            
        Raises:
            ValueError: If conversion fails
        """
        text = words.lower()
        
        # Remove parentheses, commas, and other punctuation
        text = re.sub(r"[()\[\],]", " ", text)
        text = re.sub(r"[,-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Split dollars and cents explicitly if present
        dollar_part = text
        cent_part = None
        
        # Check if text contains "cents" keyword
        if re.search(r'\bcents?\b', text, re.IGNORECASE):
            # Split on "cents" to get the part before cents
            parts = re.split(r'\s+cents?\s*$', text, flags=re.IGNORECASE)
            before_cents = parts[0].strip() if parts else text
            
            # Try to split on "and" to separate dollars and cents
            if re.search(r'\band\s+', before_cents, re.IGNORECASE):
                # Pattern: (dollar part) dollars? and (cent part)
                and_parts = re.split(r'\s+and\s+', before_cents, flags=re.IGNORECASE)
                if len(and_parts) == 2:
                    dollar_part = re.sub(r'\bdollars?\b', '', and_parts[0], flags=re.IGNORECASE).strip()
                    cent_part = and_parts[1].strip()
                else:
                    # Multiple "and"s or malformed, try to extract last part as cents
                    dollar_part = re.sub(r'\bdollars?\b', '', before_cents, flags=re.IGNORECASE).strip()
            else:
                # No "and" found, everything before "dollars" is dollar part
                dollar_part = re.sub(r'\bdollars?\b', '', before_cents, flags=re.IGNORECASE).strip()
        else:
            # No cents found, treat entire text as dollars
            # Remove "dollars" keyword if present
            dollar_part = re.sub(r'\bdollars?\b', '', text, flags=re.IGNORECASE).strip()
        
        # If dollar_part is empty or just whitespace, set to None
        if not dollar_part or not dollar_part.strip():
            dollar_part = None
        
        dollars = self._parse_integer_words(dollar_part) if dollar_part else 0
        cents = self._parse_integer_words(cent_part) if cent_part else 0
        
        if cents >= 100:
            raise ValueError(f"Cents value cannot be >= 100, got {cents} cents")
        
        result = Decimal(dollars) + (Decimal(cents) / Decimal(100))
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
    
    def _apply_enrichment_rules(self, deed_data: Dict) -> Dict:
        """
        Apply enrichment rules to calculate derived fields.
        
        Args:
            deed_data: Dictionary of deed field values
            
        Returns:
            Dictionary with enriched fields added
        """
        enriched_data = deed_data.copy()
        
        for rule in self.validation_rules.get('enrichment_rules', []):
            if rule['type'] == 'calculation':
                formula = rule['formula']
                output_field = rule['output_field']
                
                # Simple formula evaluation (replace placeholders with values)
                try:
                    # Replace field names with their values
                    eval_formula = formula
                    for key, value in deed_data.items():
                        if isinstance(value, (int, float)):
                            eval_formula = eval_formula.replace(f"{{{key}}}", str(value))
                        elif isinstance(value, datetime):
                            # Skip datetime objects in formulas
                            continue
                    
                    # Evaluate the formula safely (only basic arithmetic)
                    # Using eval is safe here because we control the input format
                    result = eval(eval_formula)
                    enriched_data[output_field] = result
                except Exception:
                    # Skip enrichment if formula evaluation fails
                    pass
        
        return enriched_data
    
    def validate(self, extracted_deed: ExtractedDeed) -> Dict:
        """
        Run all validation checks on extracted deed using rule-based system.
        
        Args:
            extracted_deed: Extracted deed data
            
        Returns:
            Dictionary with validation results and enriched data
            
        Raises:
            ValidationError: If any validation fails
        """
        # Enrich county data - use rapidfuzz scoring with LLM fallback if needed
        canonical_county, tax_rate = self.match_county(extracted_deed.county, extracted_deed.state)
        
        # Convert deed to dictionary for rule evaluation
        deed_data = {
            'doc_number': extracted_deed.doc_number,
            'county': extracted_deed.county,
            'state': extracted_deed.state,
            'date_signed': extracted_deed.date_signed,
            'date_recorded': extracted_deed.date_recorded,
            'grantor': extracted_deed.grantor,
            'grantee': extracted_deed.grantee,
            'amount_numeric': extracted_deed.amount_numeric,
            'amount_written': extracted_deed.amount_written,
            'apn': extracted_deed.apn,
            'status': extracted_deed.status,
            'tax_rate': tax_rate
        }
        
        # Apply validation rules dynamically
        for rule in self.validation_rules.get('validation_rules', []):
            rule_type = rule.get('type')
            error_type_name = rule.get('error_type')
            
            if rule_type == 'comparison':
                is_valid, error_msg = self._evaluate_comparison_rule(rule, deed_data)
                if not is_valid:
                    # Map error type names to exception classes
                    error_class_map = {
                        'DateLogicError': DateLogicError,
                        'MoneyMismatchError': MoneyMismatchError,
                        'CountyNotFoundError': CountyNotFoundError,
                        'ValidationError': ValidationError
                    }
                    error_class = error_class_map.get(error_type_name, ValidationError)
                    raise error_class(error_msg)
            
            elif rule_type == 'custom':
                is_valid, error_msg = self._evaluate_custom_rule(rule, deed_data)
                if not is_valid:
                    # Map error type names to exception classes
                    error_class_map = {
                        'DateLogicError': DateLogicError,
                        'MoneyMismatchError': MoneyMismatchError,
                        'CountyNotFoundError': CountyNotFoundError,
                        'ValidationError': ValidationError
                    }
                    error_class = error_class_map.get(error_type_name, ValidationError)
                    raise error_class(error_msg)
        
        # Apply enrichment rules
        enriched_data = self._apply_enrichment_rules(deed_data)
        
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
            'closing_costs': enriched_data.get('closing_costs', extracted_deed.amount_numeric * tax_rate)
        }
