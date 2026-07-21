"""
MediNLP symptom normalization engine.

Purpose:
- Convert natural user input into the exact symptom names used by the trained model.
- Supports English, common Banglish, selected Bangla terms, exact matching,
  generated phrase variants, fuzzy spelling matching, and simple negation handling.

This file is intentionally lightweight. It does not require online APIs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover - fallback when rapidfuzz is unavailable
    fuzz = None
    process = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:  # pragma: no cover - semantic matching is optional at runtime
    SentenceTransformer = None
    st_util = None


BANGLA_RANGE = "\u0980-\u09FF"
WORD_CHARS = rf"A-Za-z0-9_{BANGLA_RANGE}"

NEGATION_WORDS = {
    "no", "not", "without", "never", "none", "deny", "denies", "doesnt", "don't", "dont",
    "nai", "nei", "na", "nope", "নাই", "নেই", "না",
}

CONTRAST_WORDS = {"but", "kintu", "tobe", "কিন্তু", "তবে"}

# Common Banglish typo/phonetic normalization.
# This makes context phrases work even when one word is misspelled:
# "amer jhor buke batha" -> "amar jor buke betha".
BANGLISH_TOKEN_REPLACEMENTS = {
    # common non-symptom words that often appear in Banglish input
    "amer": "amar", "amr": "amar", "amiar": "amar", "amarer": "amar",
    "ase": "ache", "achee": "ache", "acce": "ache", "aca": "ache",
    "hocce": "hocche", "hoche": "hocche", "hoitese": "hocche", "hotesa": "hocche",
    "kortase": "korche", "kortese": "korche", "korce": "korche", "korchey": "korche",

    # symptom and body-part spellings
    "jhor": "jor", "zhor": "jor", "zor": "jor", "jorr": "jor", "jwr": "jor",
    "khasi": "kashi", "kasi": "kashi", "khashi": "kashi", "kashi": "kashi",
    "math": "matha", "mata": "matha", "mtha": "matha",
    "batha": "betha", "byatha": "betha", "beytha": "betha", "btha": "betha", "btha": "betha",
    "bukhe": "buke", "bukey": "buke", "buky": "buke",
    "sas": "shash", "shas": "shash", "saas": "shash", "sash": "shash",
    "kosto": "kosto", "koshto": "kosto", "ksto": "kosto", "kostho": "kosto",
    "prblm": "problem", "prob": "problem", "prblem": "problem", "problm": "problem",
    "petey": "pete", "peth": "pet", "pait": "pet",
    "bumi": "bomi", "vomi": "bomi", "vomy": "bomi", "bomi": "bomi",
    "durbol": "durbol", "durbal": "durbol", "dorbol": "durbol",
    "klanto": "klanto", "klanto": "klanto", "clanto": "klanto",
    "nakhe": "nake", "nakey": "nake", "pani": "pani",
    "prosab": "proshab", "prostab": "proshab", "pesab": "proshab", "peshab": "proshab",
    "jhapsha": "jhapsha", "japsa": "jhapsha", "jhapa": "jhapsha",
    "chulkani": "chulkani", "chulkay": "chulkani", "chulkani": "chulkani",
    "khichuni": "khichuni", "khechuni": "khichuni", "kichuni": "khichuni",
}


def normalize_banglish_typos(text: str) -> str:
    tokens = text.split()
    normalized = [BANGLISH_TOKEN_REPLACEMENTS.get(token, token) for token in tokens]
    return " ".join(normalized)

# Symptoms that must NOT be accepted only from semantic similarity.
# These symptoms can trigger red-flag alerts, so they should be detected only
# by exact dataset phrases, curated aliases, compact alias, or high fuzzy alias matching.
# This prevents false alerts such as a normal sore-throat sentence being mapped to "seizure".
SEMANTIC_BLOCKED_SYMPTOMS = {
    # Critical red flags
    "feeling suicidal", "suicidal", "homicidal thoughts",
    "unresponsiveness", "unconscious state", "stupor", "incoherent",
    "seizure", "focal seizures", "tonic seizures", "rolling of eyes", "posturing",
    "gasping for breath", "distress respiratory", "labored breathing", "cyanosis",
    "stridor", "hypoxemia", "hypercapnia", "nasal flaring",
    "st segment elevation", "st segment depression", "t wave inverted", "presence of q wave",
    "pulse absent", "cardiovascular finding cardiovascular event", "haemorrhage",
    "abdomen acute", "excruciating pain", "hypotension", "hyperkalemia",
    "hypothermia natural",

    # Major red flags
    "pain chest", "pressure chest", "chest tightness", "chest discomfort",
    "angina pectoris", "palpitation", "bradycardia",
    "sweat sweating increased", "clammy skin",
    "shortness of breath", "dyspnea", "dyspnea on exertion", "orthopnea",
    "out of breath", "catching breath", "rapid shallow breathing", "tachypnea",
    "wheezing", "breath sounds decreased", "rale", "rhonchus", "haemoptysis",
    "frothy sputum", "mental status changes", "dysarthria", "speech slurred",
    "facial paresis", "hemiplegia", "paresis", "paralyse", "paraparesis",
    "aphagia", "numbness", "numbness of hand", "syncope", "blackout",
    "syncope blackout history of blackout", "ataxia", "coordination abnormal",
    "uncoordination", "clumsiness", "vision blurred", "hemianopsia homonymous",
    "neck stiffness", "fever", "chill", "rigor temperature associated observation",
    "lethargy", "drowsiness", "sleepy", "extreme exhaustion", "malaise",
    "vomiting", "nausea and vomiting", "projectile vomiting", "hematochezia",
    "guaiac positive", "heme positive", "abdominal tenderness", "pain abdominal",
    "bowel sounds decreased", "oliguria", "hematuria", "bleeding of vagina",
    "spontaneous rupture of membranes", "fall", "dizziness", "lightheadedness",
    "orthostasis", "frail", "immobile",
}


@dataclass
class SymptomMatch:
    symptom: str
    matched_text: str
    method: str
    score: float
    status: str = "accepted"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean_text(text: Any) -> str:
    """Normalize user/model text while preserving Bangla characters."""
    value = str(text or "").replace("\xa0", " ").lower()
    value = value.replace("/", " ")
    value = re.sub(rf"[^A-Za-z0-9{BANGLA_RANGE}\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = normalize_banglish_typos(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def compact_text(text: Any) -> str:
    return clean_text(text).replace(" ", "")


def phrase_pattern(phrase: str) -> re.Pattern:
    escaped = re.escape(clean_text(phrase))
    return re.compile(rf"(?<![{WORD_CHARS}]){escaped}(?![{WORD_CHARS}])", re.IGNORECASE)


def contains_phrase(text: str, phrase: str) -> Optional[re.Match]:
    phrase = clean_text(phrase)
    if not phrase:
        return None
    return phrase_pattern(phrase).search(text)


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


def generate_ngrams(text: str, max_n: int = 6) -> List[str]:
    tokens = tokenize(text)
    phrases: List[str] = []
    for n in range(1, max_n + 1):
        for i in range(0, len(tokens) - n + 1):
            phrases.append(" ".join(tokens[i : i + n]))
    # preserve order but remove duplicates
    return list(dict.fromkeys(phrases))


def is_negated(cleaned_text: str, matched_phrase: str) -> bool:
    """
    Simple local negation detector.

    Examples blocked:
    - no cough
    - cough nai
    - fever nei
    - ami kashi na / no fever
    """
    match = contains_phrase(cleaned_text, matched_phrase)
    if not match:
        return False

    before_all = cleaned_text[: match.start()].split()
    after_tokens = cleaned_text[match.end() :].split()[:3]

    # Do not let a previous negation cross contrast words.
    # Example: "fever nai but cough ase" => fever is negated, cough is not.
    last_contrast_index = -1
    for i, token in enumerate(before_all):
        if token in CONTRAST_WORDS:
            last_contrast_index = i
    before_tokens = before_all[last_contrast_index + 1 :][-4:]

    return any(tok in NEGATION_WORDS for tok in before_tokens + after_tokens)


def fuzzy_score(a: str, b: str) -> float:
    a = clean_text(a)
    b = clean_text(b)
    if not a or not b:
        return 0.0
    if fuzz is not None:
        return float(fuzz.ratio(a, b)) / 100.0
    return SequenceMatcher(None, a, b).ratio()


def add_alias(alias_map: Dict[str, str], alias: str, symptom: str, feature_set: set) -> None:
    alias_clean = clean_text(alias)
    symptom_clean = clean_text(symptom)
    if alias_clean and symptom_clean in feature_set:
        alias_map[alias_clean] = symptom_clean


def build_manual_alias_map(feature_set: set) -> Dict[str, str]:
    """Common English/Banglish/Bangla user phrases mapped to model feature names."""
    raw_aliases: Dict[str, List[str]] = {
        "fever": [
            "fever", "fevar", "fiver", "high fever", "high temperature", "temperature", "body hot", "feel hot", "feeling hot", "hot body", "body feels hot",
            "feverish", "temperature is high", "high temp", "jor", "জ্বর", "গা গরম", "ga gorom", "gaye gorom", "gorom lagche",
        ],
        "feels hot/feverish": ["feels hot", "feverish feeling", "jor jor lagche"],
        "cough": ["cough", "caugh", "coughing", "caughing", "bad cough", "continuous cough", "dry coughing", "kashi", "khasi", "kasi", "কাশি", "কফ কাশি"],
        "productive cough": ["cough with phlegm", "cough with mucus", "kof uthche", "kof ashe", "কফ উঠছে"],
        "non-productive cough": ["dry cough", "shukna kashi", "শুকনা কাশি"],
        "headache": ["headache", "hedache", "head ache", "head pain", "my head hurts", "head hurts", "pain in head", "matha betha", "matha byatha", "মাথা ব্যথা", "মাথা বেথা"],
        "pain chest": [
            "chest pain", "pain in chest", "my chest hurts", "chest hurts", "chest hurting", "pain on chest", "buk betha", "buk byatha", "buke betha",
            "বুক ব্যথা", "বুকে ব্যথা", "বুক ব্যাথা",
        ],
        "pressure chest": ["chest pressure", "pressure in chest", "chest heaviness", "buk chap", "বুকে চাপ"],
        "chest tightness": ["tight chest", "chest tight", "buk tight", "বুক টাইট"],
        "shortness of breath": [
            "shortness of breath", "breathing problem", "difficulty breathing", "breathlessness", "hard to breathe", "hard breathing",
            "can't breathe", "cannot breathe", "shash kosto", "shash koshto", "shash nite problem",
            "shash nite koshto", "শ্বাস কষ্ট", "শ্বাস নিতে সমস্যা", "শ্বাস নিতে কষ্ট",
        ],
        "dyspnea": ["dyspnea", "dyspnoea", "breath problem"],
        "labored breathing": ["hard breathing", "breathing hard", "heavy breathing"],
        "wheezing": ["wheezing", "whistling breath", "বুকে সাঁ সাঁ", "shash e shai shai"],
        "palpitation": ["palpitation", "palpitations", "heart racing", "fast heartbeat", "buk dhorfor", "বুক ধড়ফড়"],
        "dizziness": ["dizziness", "dizzy", "matha ghurche", "মাথা ঘুরছে"],
        "lightheadedness": ["light headed", "lightheaded", "lightheadedness"],
        "vertigo": ["vertigo", "room spinning", "charpas ghure", "সব ঘুরছে"],
        "syncope": ["faint", "fainting", "passed out", "ojnan", "অজ্ঞান"],
        "nausea": ["nausea", "vomit feeling", "bomi bomi lagche", "বমি বমি লাগছে"],
        "vomiting": ["vomiting", "vomit", "throwing up", "bomi", "বমি", "bomi hocche"],
        "diarrhea": ["diarrhea", "diarrhoea", "loose motion", "loose stool", "patla paykhana", "পাতলা পায়খানা"],
        "pain abdominal": [
            "abdominal pain", "stomach pain", "belly pain", "tummy pain", "my stomach hurts", "stomach hurts", "pain in stomach", "pet betha", "pet byatha", "পেট ব্যথা",
        ],
        "abdominal bloating": ["bloating", "gas bloating", "pet fapa", "পেট ফাঁপা"],
        "heartburn": ["heartburn", "acidity", "acid reflux", "buk jala", "বুক জ্বালা"],
        "constipation": ["constipation", "hard stool", "paykhana koshte", "কোষ্ঠকাঠিন্য"],
        "fatigue": ["fatigue", "tiredness", "tired", "very tired", "exhausted", "low energy", "klanto", "ক্লান্ত"],
        "fatigue tired": ["very tired", "always tired", "extreme tiredness"],
        "asthenia": ["weakness", "weak", "body weakness", "feeling weak", "feel weak", "durbol", "দুর্বল", "shorir durbol", "body weak"],
        "malaise": ["malaise", "unwell", "not feeling well", "shorir kharap", "শরীর খারাপ"],
        "chill": ["chills", "chill", "shivering", "kapuni", "কাপুনি", "ঠান্ডা লাগা"],
        "sweat sweating increased": ["sweating", "sweat", "excessive sweating", "gham", "ঘাম"],
        "night sweat": ["night sweat", "night sweating", "rate gham", "রাতে ঘাম"],
        "pain back": ["back pain", "backache", "pith betha", "পিঠ ব্যথা"],
        "low back pain": ["lower back pain", "komor betha", "কোমর ব্যথা"],
        "pain neck": ["neck pain", "ghar betha", "ঘাড় ব্যথা"],
        "arthralgia": ["joint pain", "joint ache", "giray betha", "জয়েন্ট ব্যথা"],
        "myalgia": ["muscle pain", "muscle ache", "mangsho peshi betha", "মাংসপেশি ব্যথা"],
        "pain foot": ["foot pain", "feet pain", "pa betha", "পা ব্যথা"],
        "sore to touch": ["tender", "touch korle betha", "চাপ দিলে ব্যথা"],
        "throat sore": ["sore throat", "throat pain", "pain in throat", "throat hurts", "gola betha", "gola byatha", "গলা ব্যথা"],
        "painful swallowing": ["pain swallowing", "swallowing pain", "gilte betha", "গিলতে ব্যথা"],
        "hoarseness": ["hoarse voice", "voice change", "gola boshe geche", "গলা বসে গেছে"],
        "stuffy nose": ["stuffy nose", "blocked nose", "nak bondho", "নাক বন্ধ"],
        "nasal discharge present": ["runny nose", "nose water", "watery nose", "nasal water", "nak diye pani", "নাক দিয়ে পানি"],
        "sniffle": ["sniffle", "sniffles"],
        "snuffle": ["snuffle", "snuffles"],
        "sneeze": ["sneezing", "sneeze", "sneezes", "hachi", "হাঁচি"],
        "anosmia": ["loss of smell", "cannot smell", "smell loss", "ghran shokti nai", "ঘ্রাণ পাচ্ছি না"],
        "pruritus": ["itching", "itchy", "chulkani", "চুলকানি"],
        "erythema": ["red skin", "skin redness", "lal hoye geche", "লাল হয়ে গেছে"],
        "red blotches": ["red spots", "red patches", "lal dag", "লাল দাগ"],
        "macule": ["rash", "skin rash", "spots on skin"],
        "swelling": ["swelling", "swollen", "fola", "ফোলা"],
        "dysuria": ["burning urination", "urine burning", "jwalapora urine", "প্রসাবে জ্বালা"],
        "polyuria": ["frequent urination", "pee a lot", "bar bar prosab", "বার বার প্রসাব"],
        "hematuria": ["blood in urine", "urine blood", "প্রসাবে রক্ত"],
        "anorexia": ["loss of appetite", "no appetite", "khida nai", "খিদা নাই", "khete iccha kore na"],
        "decreased body weight": ["weight loss", "losing weight", "ojon kome jacche", "ওজন কমছে"],
        "weight gain": ["weight gain", "gaining weight", "ojon bere jacche", "ওজন বাড়ছে"],
        "vision blurred": ["blurred vision", "blurry vision", "vision problem", "chokhe jhapsha", "চোখে ঝাপসা"],
        "photophobia": ["light sensitivity", "light hurts eyes", "alo shojjo hoy na", "আলো সহ্য হয় না"],
        "tinnitus": ["ringing in ear", "ear ringing", "kane shobdo", "কানে শব্দ"],
        "numbness": ["numbness", "numb", "obosh", "অবশ"],
        "paresthesia": ["tingling", "pins and needles", "jhinjhin", "ঝিনঝিন"],
        "seizure": ["seizure", "fit", "convulsion", "khichuni", "খিঁচুনি"],
        "mood depressed": ["depressed", "depression", "mon kharap", "মন খারাপ"],
        "worry": ["worry", "worried", "chinta", "চিন্তা"],
        "anxiety": ["anxiety", "panic feeling", "anxious"],
        "sleeplessness": ["insomnia", "cannot sleep", "sleep problem", "ghum hoy na", "ঘুম হয় না"],
        "feeling suicidal": ["suicidal thoughts", "want to die", "nijeke mere felte iccha", "আত্মহত্যার চিন্তা"],
    }

    # Context-aware aliases for Banglish/Bangla/English typo combinations.
    # These are intentionally phrase-based, so a generic word like "betha" does not
    # become a symptom by itself. The nearby context word decides the model symptom.
    context_aliases: Dict[str, List[str]] = {
        "fever": [
            "jor", "jhor", "zor", "zhor", "jor ache", "jhor ache", "jor ase", "jhor ase",
            "amar jor", "amer jhor", "gaye gorom", "ga gorom", "shorir gorom", "জ্বর", "গা গরম",
        ],
        "cough": [
            "kashi", "khasi", "kasi", "kashi hocche", "khasi hocche", "কাশি", "কাশি হচ্ছে",
        ],
        "headache": [
            "matha betha", "matha batha", "mata batha", "mata betha", "math betha", "math batha",
            "matha betha korche", "মাথা ব্যথা", "মাথা ব্যাথা",
        ],
        "pain chest": [
            "buk betha", "buk batha", "buk byatha", "buke betha", "buke batha", "bukhe betha", "bukhe batha",
            "buke betha korche", "buke batha korche", "buk betha korche", "buk batha korche",
            "chest pain", "chest hurts", "my chest hurts", "বুক ব্যথা", "বুকে ব্যথা", "বুকে ব্যাথা",
        ],
        "pressure chest": [
            "buk chap", "buke chap", "bukhe chap", "chest pressure", "chest heaviness", "বুকে চাপ",
        ],
        "shortness of breath": [
            "shash kosto", "shas kosto", "sas kosto", "sash kosto", "shash nite kosto", "shas nite kosto",
            "sas nite kosto", "shash nite problem", "shas nite problem", "sas nite problem", "breath nite problem",
            "breathing problem", "hard to breathe", "difficulty breathing", "শ্বাস কষ্ট", "শ্বাস নিতে কষ্ট", "শ্বাস নিতে সমস্যা",
        ],
        "pain abdominal": [
            "pet betha", "pet batha", "pete betha", "pete batha", "pete byatha", "pet byatha",
            "stomach pain", "stomach hurts", "পেট ব্যথা", "পেটে ব্যথা", "পেট ব্যাথা",
        ],
        "vomiting": [
            "bomi", "bumi", "vomi", "bomi hocche", "bumi hocche", "vomiting", "vomit", "বমি", "বমি হচ্ছে",
        ],
        "nausea": [
            "bomi bomi", "bomi bomi lagche", "vomit feeling", "nausea", "বমি বমি", "বমি বমি লাগছে",
        ],
        "diarrhea": [
            "patla paykhana", "patla paikhana", "patla paykana", "loose motion", "loose stool", "diarrhea", "ডায়রিয়া", "পাতলা পায়খানা",
        ],
        "asthenia": [
            "durbol", "durbal", "shorir durbol", "body weak", "weakness", "দুর্বল", "শরীর দুর্বল",
        ],
        "fatigue": [
            "klanto", "khub klanto", "tired", "tiredness", "fatigue", "ক্লান্ত", "খুব ক্লান্ত",
        ],
        "dizziness": [
            "matha ghurche", "matha ghurtese", "mata ghurche", "dizzy", "dizziness", "মাথা ঘুরছে",
        ],
        "throat sore": [
            "gola betha", "gola batha", "gola byatha", "gola betha korche", "sore throat", "throat pain", "গলা ব্যথা", "গলা ব্যাথা",
        ],
        "nasal discharge present": [
            "nak diye pani", "nak diye pani pore", "nake pani", "runny nose", "nose water", "নাক দিয়ে পানি", "নাক দিয়ে পানি",
        ],
        "sneeze": ["hachi", "hasi", "sneezing", "sneeze", "হাঁচি"],
        "pruritus": ["chulkani", "chulkay", "itching", "itchy", "চুলকানি"],
        "red blotches": ["lal dag", "red spots", "red patches", "লাল দাগ"],
        "macule": ["skin rash", "rash", "skin e rash", "ত্বকে র‍্যাশ", "র‍্যাশ"],
        "dysuria": ["proshabe jala", "prosab e jala", "pesab e jala", "burning urination", "প্রসাবে জ্বালা"],
        "polyuria": ["bar bar proshab", "bar bar prosab", "bar bar pesab", "frequent urination", "বার বার প্রসাব"],
        "vision blurred": ["jhapsha dekhi", "jhapsha dekhtesi", "blurred vision", "blurry vision", "চোখে ঝাপসা", "ঝাপসা দেখছি"],
        "palpitation": ["buk dhorfor", "buke dhorfor", "heart racing", "fast heartbeat", "বুক ধড়ফড়"],
        "seizure": ["khichuni", "khechuni", "seizure", "fit", "convulsion", "খিঁচুনি"],
        "syncope": ["ojnan", "oggan", "faint", "fainting", "passed out", "অজ্ঞান"],
        "anorexia": ["khida nai", "khete iccha kore na", "loss of appetite", "no appetite", "খিদা নাই", "খেতে ইচ্ছা করছে না"],
    }
    raw_aliases.update({k: sorted(set(raw_aliases.get(k, []) + v)) for k, v in context_aliases.items()})

    alias_map: Dict[str, str] = {}
    for symptom, aliases in raw_aliases.items():
        for alias in aliases:
            add_alias(alias_map, alias, symptom, feature_set)
    return alias_map


class SymptomNormalizer:
    """Dataset-aware symptom matcher for MediNLP."""

    def __init__(
        self,
        feature_names: Iterable[str],
        enable_semantic: bool = True,
        semantic_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        semantic_threshold: float = 0.78,
    ) -> None:
        self.feature_names = [clean_text(name) for name in feature_names if clean_text(name)]
        self.feature_set = set(self.feature_names)
        self.alias_map: Dict[str, str] = {}
        self.semantic_model_name = semantic_model_name
        self.semantic_threshold = semantic_threshold
        self.semantic_enabled = False
        self.semantic_model = None
        self.semantic_terms: List[str] = []
        self.semantic_term_to_symptom: List[str] = []
        self.semantic_embeddings = None

        self._build_aliases()
        self.alias_items: List[Tuple[str, str]] = sorted(
            self.alias_map.items(), key=lambda item: len(item[0]), reverse=True
        )
        self.alias_terms: List[str] = [item[0] for item in self.alias_items]

        if enable_semantic:
            self._build_semantic_index()

    def _build_aliases(self) -> None:
        # Exact model feature names.
        for symptom in self.feature_names:
            self.alias_map[symptom] = symptom

            # Split slash alternatives: feels hot/feverish -> feels hot, feverish.
            if "/" in str(symptom):
                for part in str(symptom).split("/"):
                    add_alias(self.alias_map, part, symptom, self.feature_set)

            words = symptom.split()
            # Convert dataset style "pain chest" into user style "chest pain".
            if len(words) == 2:
                add_alias(self.alias_map, f"{words[1]} {words[0]}", symptom, self.feature_set)

            if len(words) >= 2 and words[0] in {"pain", "pressure"}:
                add_alias(self.alias_map, " ".join(words[1:] + [words[0]]), symptom, self.feature_set)

            if symptom == "throat sore":
                add_alias(self.alias_map, "sore throat", symptom, self.feature_set)

            if symptom == "sweat sweating increased":
                add_alias(self.alias_map, "excessive sweating", symptom, self.feature_set)

            if symptom == "nasal discharge present":
                add_alias(self.alias_map, "runny nose", symptom, self.feature_set)

        # Curated natural-language aliases. Do not overwrite exact model features.
        for alias, symptom in build_manual_alias_map(self.feature_set).items():
            self.alias_map.setdefault(alias, symptom)

    def _build_semantic_index(self) -> None:
        """Build a multilingual semantic index over aliases and model symptom names.

        This is optional and fail-safe. If sentence-transformers is not installed
        or the model cannot be loaded, exact/fuzzy normalization still works.
        """
        if SentenceTransformer is None or st_util is None:
            self.semantic_enabled = False
            return

        semantic_pairs: List[Tuple[str, str]] = []

        # Use all aliases plus generated natural variants as semantic labels.
        for alias, symptom in self.alias_items:
            if len(alias) >= 3:
                semantic_pairs.append((alias, symptom))

        for symptom in self.feature_names:
            words = symptom.split()
            humanized_variants = {symptom}
            if len(words) == 2:
                humanized_variants.add(f"{words[1]} {words[0]}")
            if len(words) >= 2 and words[0] in {"pain", "pressure"}:
                humanized_variants.add(" ".join(words[1:] + [words[0]]))
            if symptom.startswith("pain "):
                body_part = symptom.replace("pain ", "", 1)
                humanized_variants.add(f"{body_part} pain")
                humanized_variants.add(f"pain in {body_part}")
            if symptom.startswith("swelling "):
                body_part = symptom.replace("swelling ", "", 1)
                humanized_variants.add(f"{body_part} swelling")
                humanized_variants.add(f"swelling in {body_part}")

            for variant in humanized_variants:
                variant = clean_text(variant)
                if len(variant) >= 3:
                    semantic_pairs.append((variant, symptom))

        seen = set()
        for term, symptom in semantic_pairs:
            key = (term, symptom)
            if key in seen:
                continue
            seen.add(key)
            self.semantic_terms.append(term)
            self.semantic_term_to_symptom.append(symptom)

        try:
            self.semantic_model = SentenceTransformer(self.semantic_model_name)
            self.semantic_embeddings = self.semantic_model.encode(
                self.semantic_terms,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.semantic_enabled = True
        except Exception as exc:  # pragma: no cover - depends on local model/cache/internet
            print(f"SentenceTransformer semantic matcher disabled: {exc}")
            self.semantic_enabled = False
            self.semantic_model = None
            self.semantic_embeddings = None

    def _extract_by_semantic(
        self,
        cleaned_text: str,
        matches: Dict[str, SymptomMatch],
        negated: List[str],
    ) -> None:
        """High-confidence multilingual semantic fallback.

        No confirmation step is used. Only symptoms above the configured
        threshold are accepted; lower scores are ignored to avoid noisy medical
        predictions.
        """
        if not self.semantic_enabled or self.semantic_model is None or self.semantic_embeddings is None:
            return

        candidates = [cleaned_text] + generate_ngrams(cleaned_text, max_n=7)
        filtered_candidates: List[str] = []
        for candidate in candidates:
            candidate = clean_text(candidate)
            if len(candidate) < 4:
                continue
            candidate_tokens = set(candidate.split())
            if candidate_tokens.intersection(NEGATION_WORDS) or candidate_tokens.intersection(CONTRAST_WORDS):
                continue
            if candidate not in filtered_candidates:
                filtered_candidates.append(candidate)

        if not filtered_candidates:
            return

        try:
            candidate_embeddings = self.semantic_model.encode(
                filtered_candidates,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            score_matrix = st_util.cos_sim(candidate_embeddings, self.semantic_embeddings)
        except Exception:
            return

        for row_index, candidate in enumerate(filtered_candidates):
            row = score_matrix[row_index]
            best_score_tensor, best_index_tensor = row.max(dim=0)
            best_score = float(best_score_tensor.item())
            best_index = int(best_index_tensor.item())

            if best_score < self.semantic_threshold:
                continue

            matched_term = self.semantic_terms[best_index]
            symptom = self.semantic_term_to_symptom[best_index]

            # Never let semantic similarity alone create red-flag symptoms.
            # Red-flag symptoms are still detected through exact aliases/fuzzy aliases above.
            if symptom in SEMANTIC_BLOCKED_SYMPTOMS:
                continue

            if symptom in negated:
                continue
            if is_negated(cleaned_text, candidate) or is_negated(cleaned_text, matched_term):
                if symptom not in negated:
                    negated.append(symptom)
                continue

            # Avoid generic pain from semantic fallback if a specific pain symptom is already present.
            if symptom == "pain" and any(s != "pain" and ("pain" in s.split() or s.startswith("pain ")) for s in matches):
                continue

            self._add_match(
                matches,
                symptom,
                f"{candidate} -> {matched_term}",
                "semantic_matching",
                best_score,
            )

    def _add_match(
        self,
        matches: Dict[str, SymptomMatch],
        symptom: str,
        matched_text: str,
        method: str,
        score: float,
        status: str = "accepted",
    ) -> None:
        symptom = clean_text(symptom)
        if symptom not in self.feature_set:
            return
        current = matches.get(symptom)
        new_match = SymptomMatch(symptom=symptom, matched_text=matched_text, method=method, score=round(score, 3), status=status)
        if current is None or new_match.score > current.score:
            matches[symptom] = new_match

    def _remove_generic_overlaps(self, symptoms: List[str]) -> List[str]:
        symptom_set = set(symptoms)
        if "pain" in symptom_set and any(s != "pain" and ("pain" in s.split() or s.startswith("pain ")) for s in symptom_set):
            symptoms = [s for s in symptoms if s != "pain"]
        if "cough" in symptom_set and any(s in symptom_set for s in ["productive cough", "non productive cough", "hacking cough", "barking cough"]):
            symptoms = [s for s in symptoms if s != "cough"]
        if "fatigue" in symptom_set and "fatigue tired" in symptom_set:
            symptoms = [s for s in symptoms if s != "fatigue"]
        return symptoms

    def extract(self, message: str) -> Dict[str, Any]:
        cleaned = clean_text(message)
        accepted: Dict[str, SymptomMatch] = {}
        possible: Dict[str, SymptomMatch] = {}
        negated: List[str] = []

        if not cleaned:
            return {
                "raw_input": message,
                "cleaned_input": cleaned,
                "accepted_symptoms": [],
                "possible_symptoms": [],
                "negated_symptoms": [],
                "model_input": [],
            }

        # 1) Exact phrase + generated/manual alias matching.
        for alias, symptom in self.alias_items:
            match = contains_phrase(cleaned, alias)
            if not match:
                continue
            if is_negated(cleaned, alias):
                if symptom not in negated:
                    negated.append(symptom)
                continue
            method = "dataset_phrase" if alias == symptom else "alias_mapping"
            self._add_match(accepted, symptom, alias, method, 1.0)

        # 2) Compact matching for common Banglish variants without spaces.
        compact_input = compact_text(cleaned)
        for alias, symptom in self.alias_items:
            compact_alias = compact_text(alias)
            if len(compact_alias) < 5:
                continue
            if compact_alias in compact_input and symptom not in accepted and symptom not in negated:
                self._add_match(accepted, symptom, alias, "compact_alias", 0.97)

        # 3) Fuzzy spelling matching on ngrams.
        ngrams = generate_ngrams(cleaned, max_n=6)
        for phrase in ngrams:
            if len(phrase) < 4:
                continue
            phrase_tokens = set(phrase.split())
            if phrase_tokens.intersection(NEGATION_WORDS) or phrase_tokens.intersection(CONTRAST_WORDS):
                continue
            best_alias: Optional[str] = None
            best_symptom: Optional[str] = None
            best_score = 0.0

            if process is not None and fuzz is not None and self.alias_terms:
                extracted = process.extractOne(phrase, self.alias_terms, scorer=fuzz.ratio)
                if extracted:
                    best_alias = extracted[0]
                    best_score = float(extracted[1]) / 100.0
                    best_symptom = self.alias_map.get(best_alias)
            else:
                for alias, symptom in self.alias_items:
                    # Avoid tiny noisy comparisons.
                    if abs(len(alias) - len(phrase)) > max(8, len(alias) * 0.7):
                        continue
                    score = fuzzy_score(phrase, alias)
                    if score > best_score:
                        best_alias, best_symptom, best_score = alias, symptom, score

            if not best_alias or not best_symptom or best_symptom in negated:
                continue

            # Higher threshold for short phrases to avoid false positives.
            min_accept = 0.92 if len(phrase) <= 5 else 0.88
            min_possible = 0.86 if len(phrase) <= 5 else 0.84

            if best_score < min_possible:
                continue

            if is_negated(cleaned, phrase):
                if best_symptom not in negated:
                    negated.append(best_symptom)
                continue

            if best_score >= min_accept:
                self._add_match(accepted, best_symptom, phrase, "fuzzy_matching", best_score)
            elif len(phrase) > 4 and best_symptom not in accepted:
                self._add_match(possible, best_symptom, phrase, "fuzzy_possible", best_score, status="possible")

        # 4) SentenceTransformer semantic fallback.
        # Use semantic matching ONLY as a fallback when exact/alias/fuzzy matching
        # did not confidently find any symptom. This prevents semantic similarity
        # from adding extra false red-flag symptoms to normal Bangla/Banglish inputs
        # such as fever + cough + headache.
        if not accepted:
            self._extract_by_semantic(cleaned, accepted, negated)

        accepted_list = sorted(accepted.values(), key=lambda m: (-m.score, m.symptom))
        possible_list = [m for symptom, m in possible.items() if symptom not in accepted]
        possible_list = sorted(possible_list, key=lambda m: (-m.score, m.symptom))[:10]

        accepted_symptoms = self._remove_generic_overlaps([m.symptom for m in accepted_list])
        accepted_list = [m for m in accepted_list if m.symptom in accepted_symptoms]

        return {
            "raw_input": message,
            "cleaned_input": cleaned,
            "accepted_symptoms": [m.to_dict() for m in accepted_list],
            "possible_symptoms": [m.to_dict() for m in possible_list],
            "negated_symptoms": sorted(negated),
            "model_input": accepted_symptoms,
            "semantic_enabled": self.semantic_enabled,
            "semantic_threshold": self.semantic_threshold,
        }
