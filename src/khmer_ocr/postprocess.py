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
        'ន.ស.': 'នាយកសាលា',
        'ប.ស.': 'បណ្ឌិតសភា',
        'ព.ស.': 'ពុទ្ធសករាជ',
        'ច.ក.': 'ចក្រភព',
        'ស.ក.': 'សាធារណៈកម្ពុជា',
        'ក.រ.': 'ក្រសួងរាជធានី',
        'ក.ស.': 'ក្រសួងសាធារណការ',
    }
    
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    return text


def spell_check_khmer(text, custom_dict=None):
    """Correct common spelling errors in Khmer text"""
    common_error = {
        'មហាុ': 'មហា',
        'ប្រទេសជ': 'ប្រទេស',
        'កមពុជ': 'កម្ពុជា',
        'បរជាតិយ': 'ប្រជាជាតិ',
        'សកលវិទ្យាលយ': 'សកលវិទ្យាល័យ',
        'អង្គរវត្ត្': 'អង្គរវត្ត',
        'បាំបាំ': 'ប៉ាប៉ា',
        'ព្រះរាជា': 'ព្រះរាជា',   
        'ពរះ': 'ព្រះ',
        'ព្រះរាជធានីភញ': 'ព្រះរាជធានីភ្នំពេញ',
        'ភញ': 'ភ្នំពេញ',
        'ខមរភាសា': 'ខ្មែរ​ភាសា',
        'ភាសាខមរ': 'ភាសាខ្មែរ',
        'កមពុជះ': 'កម្ពុជា',
        'កំពុជា': 'កម្ពុជា',
        'កមពុជា': 'កម្ពុជា',
        'អន្ដរជាត': 'អន្តរជាតិ',
        'អន្ដរជាតិ': 'អន្តរជាតិ',
        'បរទេស្': 'បរទេស',
        'នយោបាយ្': 'នយោបាយ',
        'រដ្ឋបាល្': 'រដ្ឋបាល',
        'សេដ្ឋកិច': 'សេដ្ឋកិច្ច',
        'សេដកិច្ច': 'សេដ្ឋកិច្ច',
        'សិល្បៈ្': 'សិល្បៈ',
        'វប្បធម៌្': 'វប្បធម៌',
        'វិទ្យាលយ': 'វិទ្យាល័យ',
        'សាកលវិទ្យាលយ': 'សាកលវិទ្យាល័យ',
        'សាលាវទ្យាល័យ': 'សាលាវិទ្យាល័យ',
        'សាលាផ្លូវមធ្យម': 'សាលាផ្លូវមធ្យមសិក្សា',
        'រាជធានីភញ': 'រាជធានីភ្នំពេញ',
        'សុខាភិបាល្': 'សុខាភិបាល',
        'គមនាគមន៍្': 'គមនាគមន៍',
        'ធនធានមនុស្ស្': 'ធនធានមនុស្ស',
        'អប់រំ្': 'អប់រំ',
        'កសិកម្ម្': 'កសិកម្ម',
        'ឧស្សាហកម្ម្': 'ឧស្សាហកម្ម',
        'ពាណិជ្ជកម្ម': 'ពាណិជ្ជកម្ម',
        'ជ្ជកម្ម': 'ជួញជុល',  
        'សន្តិសុខ្': 'សន្តិសុខ',
        'ជាតិខមរ': 'ជាតិខ្មែរ',
        'ប្រជាជាតខមរ': 'ប្រជាជាតិខ្មែរ',
        'ក្រសួងអប់រំ្': 'ក្រសួងអប់រំ',
        'ក្រសួងសុខាភិបាល្': 'ក្រសួងសុខាភិបាល',
        'សាលារៀន្': 'សាលារៀន',
        'ព្រះសង្ឃ្': 'ព្រះសង្ឃ',
        'ព្រះបរមរាជវង្ស្': 'ព្រះបរមរាជវង្ស',
        'សិទ្ធមនុស្ស្': 'សិទ្ធិមនុស្ស',
        'មនោសញ្ចេតនា': 'មនោសញ្ចេតនា',
        'សហរដ្ឋអាមេរិក្': 'សហរដ្ឋអាមេរិក',
        'បារាំង្': 'បារាំង',
        'ចិន្': 'ចិន',
        'ថៃ្': 'ថៃ',
    }

    for error, correction in common_error.items():
        text = text.replace(error, correction)
    return text


def postprocess_pipeline(text, preserve_lines=False):
    """Complete post-processing pipeline for Khmer text"""
    # Apply normalization steps
    text = normalize_khmer_unicode(text)
    text = correct_khmer_spacing(text)
    text = numbers_to_khmer(text)
    text = expand_khmer_abbreviations(text)
    text = spell_check_khmer(text)

    if preserve_lines:
        lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    # Additional cleaning
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()

    return text
