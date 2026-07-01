import re
from typing import Dict, Optional

# Standard mapping dictionary from various formats to canonical names
# Canonical names are based on martj42's international results dataset.
STANDARD_MAP = {
    # Americas
    "USA": "United States",
    "United States of America": "United States",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "U.S. Virgin Islands": "United States Virgin Islands",
    "US Virgin Islands": "United States Virgin Islands",
    "Curacao": "Curaçao",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "St. Lucia": "Saint Lucia",
    "Saint Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "Saint Kitts / Nevis": "Saint Kitts and Nevis",

    # Asia
    "Korea Republic": "South Korea",
    "Korea, South": "South Korea",
    "Korea": "South Korea",  # default to South Korea if just 'Korea' in some contexts
    "Korea DPR": "North Korea",
    "Korea, North": "North Korea",
    "China PR": "China",
    "IR Iran": "Iran",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Brunei Darussalam": "Brunei",
    "Palestine": "State of Palestine",
    "East Timor": "Timor-Leste",
    "Chinese Taipei": "Taiwan",
    "Macau": "Macao",

    # Europe
    "Czechia": "Czech Republic",
    "Macedonia": "North Macedonia",
    "FYR Macedonia": "North Macedonia",
    "Republic of Ireland": "Eire",  # check results.csv naming
    "Irish Republic": "Eire",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Germany FR": "Germany FR",  # keep separate if results.csv has Germany FR/West Germany
    "German DR": "East Germany",
    "Germany DR": "East Germany",
    "Soviet Union": "Soviet Union",
    "Yugoslavia": "Yugoslavia",
    "Czechoslovakia": "Czechoslovakia",

    # Africa
    "Congo DR": "DR Congo",
    "Congo, Democratic Republic of the": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Congo, Democratic Republic of": "DR Congo",
    "Zaire": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Swaziland": "Eswatini",
}

def clean_name(name: Optional[str]) -> str:
    """Basic cleaning of name strings: strip whitespace and normalize case/accents."""
    if not isinstance(name, str):
        return ""
    # Remove leading/trailing whitespace
    name = name.strip()
    # Replace multiple spaces with a single space
    name = re.sub(r'\s+', ' ', name)
    return name

def standardize_team_name(name: str) -> str:
    """Maps a raw team name to its canonical form using the standard map."""
    cleaned = clean_name(name)
    if not cleaned:
        return ""
    
    # 1. Check exact match in standard map
    if cleaned in STANDARD_MAP:
        return STANDARD_MAP[cleaned]
        
    # 2. Check case-insensitive match in standard map
    for k, v in STANDARD_MAP.items():
        if k.lower() == cleaned.lower():
            return v
            
    # 3. Handle common naming patterns
    # e.g., "Viet Nam" -> "Vietnam"
    if cleaned.lower() == "viet nam":
        return "Vietnam"
        
    return cleaned

def get_mapping_table() -> Dict[str, str]:
    """Returns the name-mapping dictionary for debugging/display."""
    return STANDARD_MAP
