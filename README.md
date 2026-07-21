# MediNLP Web Chatbot

MediNLP is a FastAPI + React medical symptom chatbot built from the trained MediNLP notebook pipeline.

## Main flow

1. User writes symptoms in normal language.
2. Backend normalizes the text into the exact dataset/model symptom names.
3. Red-flag screening runs first.
4. If no red flag is found, disease prediction runs.
5. The app shows Top-3 predicted diseases.
6. SHAP/explainability is shown when enabled.
7. Doctor type, suggested tests, care note, and urgency are shown.
8. The app provides a Google Maps “near me” link for the recommended specialist/hospital.

## Input normalization

This version includes dataset-aware symptom normalization through:

- exact model symptom matching
- automatic phrase variants, e.g. `chest pain` → `pain chest`
- English aliases
- Banglish/Bangla aliases, e.g. `jor`, `kashi`, `buk betha`
- fuzzy typo handling, e.g. `fevar`, `caugh`
- simple negation handling, e.g. `fever nai but cough ase` → only `cough`

The normalizer is in:

```text
backend/symptom_engine.py
```

It uses the actual model feature list from:

```text
backend/model_files/symptom_feature_list.csv
```

## Required files in `backend/model_files/`

- `final_multinomial_nb_model.joblib`
- `feature_names.joblib`
- `disease_classes.joblib`
- `symptom_feature_list.csv`
- `model_disease_classes.csv`
- `disease_recommendation_lookup_minimal.json`
- `red_flag_rules.json`

## Backend setup

```bash
cd backend

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually:

```text
http://127.0.0.1:5173
```

## Nearby doctor / hospital search

This project does **not** use a backend Hospital API, Foursquare, or Overpass.

The frontend uses a simple Google Maps text search link after prediction:

```text
Recommended specialist + "hospital near me"
```

Example:

```text
Cardiologist hospital near me
Pulmonologist hospital near me
Medicine Specialist hospital near me
```

No Google Maps API key is required for this link-based approach.

## Test examples

Try these inputs in the chatbot:

```text
amar jor ase, kashi hocche, matha betha
amar buk betha ar shash nite problem hocche
fevar and caugh hocche
amar fever nai but cough ase
pet betha ar patla paykhana
```

## API health check

```bash
curl http://127.0.0.1:8000/api/health
```

The response includes:

```json
{
  "symptom_normalizer_loaded": true,
  "nearby_doctor_search": "Google Maps text link only; no Hospital API/Foursquare/Overpass backend"
}
```
