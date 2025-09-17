"""
NBA-specific tokenization utilities for better model performance.
Handles team names, basketball terminology, and number formatting.
"""

import re
from typing import List, Dict, Any
from loguru import logger


# Complete NBA team mappings
NBA_TEAMS = {
    "ATL": "Hawks",
    "BRK": "Nets", 
    "BOS": "Celtics",
    "CHO": "Hornets",
    "CHI": "Bulls",
    "CLE": "Cavaliers",
    "DAL": "Mavericks",
    "DEN": "Nuggets",
    "DET": "Pistons",
    "GSW": "Warriors",
    "HOU": "Rockets",
    "IND": "Pacers",
    "LAC": "Clippers",
    "LAL": "Lakers",
    "MEM": "Grizzlies",
    "MIA": "Heat",
    "MIL": "Bucks",
    "MIN": "Timberwolves",
    "NOP": "Pelicans",
    "NYK": "Knicks",
    "OKC": "Thunder",
    "ORL": "Magic",
    "PHI": "76ers",
    "PHO": "Suns",
    "POR": "Trail Blazers",
    "SAC": "Kings",
    "SAS": "Spurs",
    "TOR": "Raptors",
    "UTA": "Jazz",
    "WAS": "Wizards"
}

# All team names (abbreviations + full names)
ALL_TEAM_NAMES = list(NBA_TEAMS.keys()) + list(NBA_TEAMS.values())

# Basketball-specific terminology that should be single tokens
BASKETBALL_TERMS = [
    # Shot types
    "3-pointer", "3-pointers", "3s", "3-point", "3-points",
    "2-pointer", "2-pointers", "2s", "2-point", "2-points", 
    "free throw", "free throws", "foul shot", "foul shots",
    "layup", "layups", "dunk", "dunks", "jumper", "jumpers",
    "bank shot", "bank shots", "finger roll", "finger rolls",
    "step-back", "step-back move", "alley-oop", "alley-oops",
    
    # Statistics
    "rebound", "rebounds", "assist", "assists", "steal", "steals",
    "block", "blocks", "turnover", "turnovers", "foul", "fouls",
    "double-double", "triple-double", "career high", "season high",
    
    # Game events
    "technical foul", "technical fouls", "flagrant foul", "flagrant fouls",
    "ejection", "ejections", "suspension", "suspensions",
    "timeout", "timeouts", "overtime", "overtimes",
    
    # Positions and roles
    "point guard", "shooting guard", "small forward", "power forward", "center",
    "PG", "SG", "SF", "PF", "C", "coach", "coaches", "referee", "referees",
    
    # Game situations
    "playoff", "playoffs", "postseason", "regular season", "finals",
    "conference", "eastern conference", "western conference",
    "first round", "second round", "conference finals", "nba finals",
    
    # Common abbreviations
    "NBA", "MVP", "DPOY", "ROY", "6MOY", "MIP", "COY",
    "All-Star", "All-NBA", "All-Defensive", "All-Rookie"
]

# Note: Number patterns are handled directly in preprocess_text() function

def get_custom_tokens() -> List[str]:
    """Get all custom tokens that should be added to the tokenizer."""
    return ALL_TEAM_NAMES + BASKETBALL_TERMS

def preprocess_text(text: str) -> str:
    """
    Preprocess text to make it more tokenizer-friendly.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    logger.debug(f"Preprocessing text: {text[:100]}...")
    
    # Convert scores from "117-109" to "117 to 109" for better tokenization
    text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)
    
    # Convert "3-pointer" to "three pointer" for better tokenization
    text = re.sub(r'(\d+)-pointer', r'\1 pointer', text)
    text = re.sub(r'(\d+)-point', r'\1 point', text)
    
    # Convert "3s" to "three pointers" for consistency
    text = re.sub(r'\b(\d+)s\b', r'\1 pointers', text)
    
    # Convert time formats to be more readable
    text = re.sub(r'(\d+):(\d+)', r'\1 minutes \2 seconds', text)
    
    # Convert ratios to be more readable
    text = re.sub(r'(\d+)/(\d+)', r'\1 out of \2', text)
    
    # Convert percentages to be more readable
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    
    logger.debug(f"Preprocessed text: {text[:100]}...")
    return text

def postprocess_text(text: str) -> str:
    """
    Postprocess generated text to restore proper formatting.
    
    Args:
        text: Generated text to postprocess
        
    Returns:
        Postprocessed text
    """
    logger.debug(f"Postprocessing text: {text[:100]}...")
    
    # Restore scores from "117 to 109" to "117-109"
    text = re.sub(r'(\d+) to (\d+)', r'\1-\2', text)
    
    # Restore "three pointer" to "3-pointer"
    text = re.sub(r'three pointer', '3-pointer', text)
    text = re.sub(r'three point', '3-point', text)
    text = re.sub(r'two pointer', '2-pointer', text)
    text = re.sub(r'two point', '2-point', text)
    
    # Restore "three pointers" to "3-pointers"
    text = re.sub(r'three pointers', '3-pointers', text)
    text = re.sub(r'two pointers', '2-pointers', text)
    
    # Restore time formats
    text = re.sub(r'(\d+) minutes (\d+) seconds', r'\1:\2', text)
    
    # Restore ratios
    text = re.sub(r'(\d+) out of (\d+)', r'\1/\2', text)
    
    # Restore percentages
    text = re.sub(r'(\d+) percent', r'\1%', text)
    
    logger.debug(f"Postprocessed text: {text[:100]}...")
    return text

def add_custom_tokens_to_tokenizer(tokenizer: Any) -> Any:
    """
    Add custom tokens to a tokenizer.
    
    Args:
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Updated tokenizer with custom tokens
    """
    custom_tokens = get_custom_tokens()
    
    # Add tokens that don't exist in the vocabulary
    new_tokens = []
    for token in custom_tokens:
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)
    
    if new_tokens:
        logger.info(f"Adding {len(new_tokens)} custom tokens to tokenizer")
        tokenizer.add_tokens(new_tokens)
        logger.debug(f"Added tokens: {new_tokens[:10]}...")  # Log first 10
    else:
        logger.info("No new tokens to add to tokenizer")
    
    return tokenizer

def get_team_name_mapping() -> Dict[str, str]:
    """Get the mapping from abbreviations to full team names."""
    return NBA_TEAMS.copy()

def get_team_abbreviation_mapping() -> Dict[str, str]:
    """Get the mapping from full team names to abbreviations."""
    return {v: k for k, v in NBA_TEAMS.items()}

def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name to full name.
    
    Args:
        team_name: Team name (abbreviation or full name)
        
    Returns:
        Full team name
    """
    team_name = team_name.upper()
    
    # If it's an abbreviation, return full name
    if team_name in NBA_TEAMS:
        return NBA_TEAMS[team_name]
    
    # If it's already a full name, return as is
    if team_name in NBA_TEAMS.values():
        return team_name
    
    # Try to find by partial match
    for abbrev, full_name in NBA_TEAMS.items():
        if team_name in full_name.upper() or full_name.upper() in team_name:
            return full_name
    
    logger.warning(f"Unknown team name: {team_name}")
    return team_name
