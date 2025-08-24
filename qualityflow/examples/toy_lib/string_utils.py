"""
String utility functions for QualityFlow demonstration.
"""

import re
from typing import List, Optional


def reverse_string(s: str) -> str:
    """Reverse a string."""
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    return s[::-1]


def is_palindrome(s: str, ignore_case: bool = True) -> bool:
    """Check if a string is a palindrome."""
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    
    # Clean the string - keep only alphanumeric characters
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s)
    
    if ignore_case:
        cleaned = cleaned.lower()
    
    return cleaned == cleaned[::-1]


def count_words(text: str) -> int:
    """Count words in text."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    if not text.strip():
        return 0
    
    words = text.split()
    return len(words)


def capitalize_words(text: str) -> str:
    """Capitalize the first letter of each word."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    return ' '.join(word.capitalize() for word in text.split())


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to maximum length with suffix."""
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    if not isinstance(max_length, int) or max_length < 0:
        raise ValueError("max_length must be a non-negative integer")
    
    if len(s) <= max_length:
        return s
    
    if max_length <= len(suffix):
        return s[:max_length]
    
    return s[:max_length - len(suffix)] + suffix


class TextProcessor:
    """Text processing utility class."""
    
    def __init__(self, default_encoding: str = "utf-8"):
        self.default_encoding = default_encoding
        self.processed_count = 0
    
    def clean_text(self, text: str, remove_punctuation: bool = False) -> str:
        """Clean text by removing extra whitespace and optionally punctuation."""
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        if remove_punctuation:
            # Remove punctuation except spaces
            cleaned = re.sub(r'[^\w\s]', '', cleaned)
        
        self.processed_count += 1
        return cleaned
    
    def word_frequency(self, text: str, ignore_case: bool = True) -> dict[str, int]:
        """Count word frequency in text."""
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        
        words = text.split()
        if ignore_case:
            words = [word.lower() for word in words]
        
        frequency = {}
        for word in words:
            # Remove punctuation from word
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word:
                frequency[clean_word] = frequency.get(clean_word, 0) + 1
        
        return frequency
    
    def get_stats(self) -> dict[str, int]:
        """Get processing statistics."""
        return {
            "processed_count": self.processed_count,
            "default_encoding": self.default_encoding
        }