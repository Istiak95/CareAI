# MediNLP Input Normalization Update

## Added

- `backend/symptom_engine.py`
- Dataset-aware symptom normalization before red-flag screening and prediction
- English/Banglish/Bangla alias support
- Fuzzy spelling support through `rapidfuzz`
- Negation handling
- API response fields for normalization details:
  - `extracted_symptoms`
  - `possible_symptoms`
  - `negated_symptoms`
  - `symptom_extraction`

## Modified

- `backend/main.py`
  - Replaced the old exact-only extractor with the new hybrid normalizer.
  - `/api/chat` now converts natural user text into model feature names first.
  - `/api/health` now reports normalizer status.

- `backend/requirements.txt`
  - Added `rapidfuzz`.
  - Removed unused `httpx`.

- `frontend/src/App.jsx`
  - Updated starter message, tips, and placeholder for natural input.
  - Shows detected symptoms in the result summary.
  - Keeps the existing Google Maps near-me specialist link.

- `frontend/package.json`
  - Removed unused `leaflet` dependency.

## Removed

The project now uses only the Google Maps text-link approach for nearby doctor/hospital search. These Foursquare/Overpass/backend hospital API files were removed:

- `FOURSQUARE_SETUP.md`
- `HOSPITAL_FEATURE.md`
- `CODE_CHANGES.md`
- `QUICK_START.md`
- `backend/FOURSQUARE_DEBUG.md`
- `backend/test_foursquare.py`

## Example mappings

```text
amar jor ase, kashi hocche, matha betha
→ fever, cough, headache

amar buk betha ar shash nite problem hocche
→ pain chest, shortness of breath

fevar and caugh hocche
→ fever, cough

amar fever nai but cough ase
→ cough
```

## Latest advanced normalization update

Added context-aware Banglish typo normalization and phrase matching.
Examples:

- `amer jhor buke batha` → `fever`, `pain chest`
- `amar khasi ar math batha` → `cough`, `headache`
- `amar sas nite prblm hocche` → `shortness of breath`
- `amar pete batha ar bumi hocche` → `pain abdominal`, `vomiting`

The normalizer still supports English, Bangla, Banglish, fuzzy matching, and optional SentenceTransformer semantic fallback.
