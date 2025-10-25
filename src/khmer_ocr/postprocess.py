"""
Post-processing for Khmer text
Handles Unicode normalization, spacing correction, and error correction
"""
import unicodedata
import re


def normalize_khmer_unicode(text):
    """Normalize Khmer Unicode characters"""
    normalized = unicodedata.normalize('NFC', text)
    error_corrections = {
        'ាឹ': 'ា',
        'ិះ': 'ិ',
        'ុះ': 'ុ',
        '៉ា': 'ា',
    }
    for error, correction in error_corrections.items():
        normalized = normalized.replace(error, correction)
    return normalized


def correct_khmer_spacing(text):
    """Correct spacing issues in Khmer text"""
    khmer_pattern = r'([\u1780-\u17FF]+)\s+([\u1780-\u17FF]+)'
    corrected = re.sub(khmer_pattern, r'\1\2', text)
    return corrected.strip()


def numbers_to_khmer(text):
    """Convert Arabic numerals to Khmer numerals"""
    arabic_digits = '0123456789'
    khmer_digits = '០១២៣៤៥៦៧៨៩'
    translation_table = str.maketrans(arabic_digits, khmer_digits)
    return text.translate(translation_table)


def expand_khmer_abbreviations(text):
    """Expand common Khmer abbreviations"""
    abbreviations = {
        'គ.ស.': 'គ្រិស្តសករាជ',
        'ម.រ.': 'មុនគ្រិស្តសករាជ',
        'រ.ដ.': 'រដ្ឋាភិបាល',
        'ឯ.អ.': 'ឯកអគ្គ',
        'អ.ដ.': 'អនុដ្ឋាន',
        'ស.រ.': 'សាធារណរដ្ឋ',
    }
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    return text


def spell_check_khmer(text, custom_dict=None):
    """Correct common spelling errors in Khmer text"""
    common_errors = {
        'មហាុ': 'មហា',
        'ប្រទេសជ': 'ប្រទេស',
        'កមពុជ': 'កម្ពុជា',
        'បរជាតិយ': 'ប្រជាជាតិ',
        'សកលវិទ្យាលយ': 'សកលវិទ្យាល័យ',
        'អង្គរវត្ត្': 'អង្គរវត្ត',
        'បាំបាំ': 'ប៉ាប៉ា',
    }
    for error, correction in common_errors.items():
        text = text.replace(error, correction)
    return text


def postprocess_pipeline(text):
    """Complete post-processing pipeline for Khmer text"""
    # Apply normalization steps
    text = normalize_khmer_unicode(text)
    text = correct_khmer_spacing(text)
    text = numbers_to_khmer(text)
    text = expand_khmer_abbreviations(text)
    text = spell_check_khmer(text)
    
    # Additional cleaning
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    return text
