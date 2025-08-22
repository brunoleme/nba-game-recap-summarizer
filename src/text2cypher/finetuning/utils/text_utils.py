import re

def replace_first_dash(text: str) -> str:
    """
    Replace the first dash with a slash if there are 3 or more dashes.
    Useful for transforming model names into paths.
    Example: 't5-small-notechat-512' -> 't5/small-notechat-512'
    """
    if text.count("-") >= 3:
        return text.replace("-", "/", 1)
    return text

def get_first_digit(s: str) -> int:
    """
    Extract the first digit from a string.
    Example: 'v3.2.1' -> 3
    """
    match = re.search(r'\d', s)
    return int(match.group()) if match else None

def clean_conversation(text: str) -> str:
    """
    Clean and normalize a clinical conversation string.
    - Removes control characters (except newline/tab)
    - Normalizes quotes
    - Replaces excessive whitespace
    """
    text = re.sub(r"[\x00-\x09\x0B-\x1F\x7F]+", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text
