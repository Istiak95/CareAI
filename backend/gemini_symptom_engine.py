from __future__ import annotations

import os
import json
from typing import List, Optional, Literal

from google import genai
from pydantic import BaseModel, Field


class GeminiSymptomResult(BaseModel):

    language: Literal[
        "english",
        "bangla",
        "banglish",
        "mixed",
        "unknown",
    ] = "unknown"

    # What kind of message did the user send?
    intent: Literal[
        "symptom_input",
        "symptom_followup",
        "explanation_request",
        "greeting",
        "other",
    ] = "symptom_input"

    intent_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    canonical_symptoms: List[str] = Field(
        default_factory=list
    )

    negated_symptoms: List[str] = Field(
        default_factory=list
    )

    clarification_needed: bool = False

    follow_up_question: Optional[str] = None


class GeminiSymptomEngine:

    def __init__(
        self,
        feature_names: List[str],
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):

        self.feature_names = sorted(
            {
                str(symptom)
                .strip()
                .lower()

                for symptom
                in feature_names

                if str(symptom).strip()
            }
        )

        self.feature_set = set(
            self.feature_names
        )

        self.api_key = (
            api_key
            or os.getenv(
                "GEMINI_API_KEY",
                "",
            )
        ).strip()

        self.model = (
            model
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-3.6-flash",
            )
        ).strip()

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing."
            )

        # Google GenAI SDK reads this environment variable.
        os.environ[
            "GEMINI_API_KEY"
        ] = self.api_key

        self.client = genai.Client()

        self.allowed_symptoms_text = (
            "\n".join(
                f"- {symptom}"
                for symptom
                in self.feature_names
            )
        )

    def explain_previous_result(
        self,
        previous_result,
        language: str,
        user_message: str = "",
    ):
        """
        Explain CareAI's PREVIOUS model result.

        This never runs a new disease prediction.
        """

        if not isinstance(previous_result, dict):
            return None

        predictions = []

        for pred in (
            previous_result.get(
                "top_predictions"
            )
            or []
        )[:3]:

            shap_items = (
                (
                    pred.get(
                        "shap_explanation"
                    )
                    or {}
                ).get(
                    "present_symptom_contributions"
                )
                or []
            )

            predictions.append(
                {
                    "rank":
                        pred.get("rank"),

                    "disease":
                        pred.get("disease"),

                    "confidence_percent":
                        pred.get(
                            "confidence_percent"
                        ),

                    "shap":
                        shap_items[:5],

                    "recommendation":
                        pred.get(
                            "recommendation"
                        )
                        or {},
                }
            )

        red_flag_result = (
            previous_result.get(
                "red_flag_result"
            )
            or {}
        )

        summary = {
            "status":
                previous_result.get(
                    "status"
                ),

            "red_flag":
                previous_result.get(
                    "red_flag",
                    False,
                ),

            "matched_symptoms":
                previous_result.get(
                    "matched_symptoms",
                    [],
                ),

            "negated_symptoms":
                previous_result.get(
                    "negated_symptoms",
                    [],
                ),

            "red_flag_result":
                {
                    "severity":
                        red_flag_result.get(
                            "severity"
                        ),

                    "reason":
                        red_flag_result.get(
                            "reason"
                        ),

                    "triggered_symptoms":
                        red_flag_result.get(
                            "triggered_symptoms",
                            [],
                        ),
                },

            "top_predictions":
                predictions,
        }

        if language == "bangla":

            language_instruction = """
Write the explanation in natural, easy
BANGLA SCRIPT.

The user may have asked in Banglish,
but your explanation MUST be in Bangla.

Disease names, doctor types, test names,
SHAP, and percentages may remain English
when that is clearer.

Do NOT reply in Banglish.
"""

        else:

            language_instruction = """
Write the explanation in clear, easy English.
Do not switch to Bangla or Banglish.
"""

        prompt = f"""
You are CareAI's explanation assistant.

The user is asking you to explain an ALREADY
GENERATED CareAI result.

IMPORTANT RULES:

- Do NOT make a new disease prediction.
- Do NOT change any confidence score.
- Do NOT invent symptoms.
- Do NOT invent SHAP values.
- Do NOT diagnose the user.
- Do NOT prescribe medicine.
- Explain only the supplied CareAI result.
- Clearly say that model predictions are not
  confirmed medical diagnoses.
- Explain confidence in simple language.
- If SHAP information exists, explain which
  PRESENT symptoms influenced the model score.
- SHAP influence is model influence, NOT proof
  that a symptom caused a disease.
- If this is a red-flag result, explain why the
  safety warning appeared and preserve its urgency.
- Keep the explanation useful and reasonably short.

{language_instruction}

USER'S EXPLANATION REQUEST:
{user_message}

PREVIOUS CAREAI RESULT:
{json.dumps(summary, ensure_ascii=False)}
"""

        try:

            interaction = (
                self.client
                .interactions
                .create(
                    model=self.model,
                    input=prompt,
                    store=False,
                )
            )

            explanation = str(
                interaction.output_text
                or ""
            ).strip()

            return explanation or None

        except Exception as exc:

            print(
                "Gemini explanation failed:",
                exc,
            )

            return None


    def analyze(
        self,
        message: str,
    ):

        message = str(
            message or ""
        ).strip()

        if not message:
            return None

        prompt = f"""
You are the symptom-understanding NLP layer
for CareAI.

IMPORTANT:

You are NOT the disease prediction model.

Do NOT diagnose diseases.

Do NOT recommend medicines.

Do NOT provide treatment.

Your task is ONLY to understand symptom text.

The user may write in:

- English
- Bangla
- Banglish
- mixed Bangla/English
- spelling mistakes
- informal language
- phonetic spelling

TASKS:

1. Detect whether the input language is:
   english, bangla, banglish, mixed, or unknown.

2. Understand spelling mistakes.

3. Understand Banglish and Bangla naturally.

4. Semantically understand symptom descriptions.

5. Map clear symptoms ONLY to the exact
   canonical symptom names from the allowed
   symptom list below.

6. NEVER invent a new symptom.

7. Every canonical symptom MUST exactly match
   one of the allowed symptom names.

INTENT CLASSIFICATION:

Before extracting symptoms, classify what
the user is actually trying to do.

Use exactly one intent:

symptom_input
= the user is describing one or more
  current symptoms.

symptom_followup
= the user is answering a previous
  clarification question with more
  symptom information.

explanation_request
= the user is asking to understand,
  explain, simplify, or clarify an
  earlier result or message.

Examples:
"amk bujai dao eta"
"eta bujhlam na"
"explain this"
"aro easy kore bolo"
"এটা বুঝিয়ে বলো"

These are NOT new symptom descriptions.

greeting
= hello, hi, thanks, etc.

other
= unrelated/non-medical text or text
  that is not describing symptoms.

VERY IMPORTANT:

If intent is explanation_request,
greeting, or other:

- canonical_symptoms MUST be []
- negated_symptoms MUST be []
- clarification_needed MUST be false
- NEVER guess symptoms from semantic
  similarity.

For symptom_input and symptom_followup,
continue normal symptom extraction.

Return a confidence from 0.0 to 1.0
in intent_confidence.

NEGATION:

Understand negation in English, Bangla,
and Banglish.

Negation scope is VERY IMPORTANT.

"nai", "nei", "na", "no", and "not"
must apply only to the symptom phrase
they actually describe.

Example 1:

"jor ase kashi nai"

fever = present symptom
cough = negated symptom

Therefore:

canonical_symptoms = ["fever"]
negated_symptoms = ["cough"]

Example 2:

"jor nai kashi ase"

fever = negated symptom
cough = present symptom

Therefore:

canonical_symptoms = ["cough"]
negated_symptoms = ["fever"]

Example 3:

"amar jor nai but kashi ase"

fever = negated symptom
cough = present symptom

Never negate an earlier symptom only because
a later symptom is followed by "nai", "nei",
or "na".

Also remember that Banglish "ase" usually
means the Bangla verb "আছে" (is/has).
Do NOT interpret "ase" as the English
symptom "ache".

A negated symptom MUST NOT be placed inside
canonical_symptoms.

CLARIFICATION:

If NO symptom can safely be identified,
but the user appears to be describing a
symptom, do not guess.

Set:

clarification_needed = true

Then generate ONE short follow_up_question.

Use the same language style as the user.

Examples:

Banglish user -> Banglish question

Bangla user -> Bangla question

English user -> English question

If the text is unrelated or meaningless,
return no symptoms.

ALLOWED CANONICAL SYMPTOMS:

{self.allowed_symptoms_text}

USER MESSAGE:

{message}
"""

        try:

            interaction = (
                self.client
                .interactions
                .create(
                    model=self.model,

                    input=prompt,

                    # Do not keep this symptom
                    # request as a Gemini
                    # conversation state.
                    store=False,

                    response_format={
                        "type":
                            "text",

                        "mime_type":
                            "application/json",

                        "schema":
                            GeminiSymptomResult
                            .model_json_schema(),
                    },
                )
            )

            parsed = (
                GeminiSymptomResult
                .model_validate_json(
                    interaction.output_text
                )
            )

            canonical = []

            for symptom in (
                parsed.canonical_symptoms
            ):

                symptom = str(
                    symptom
                ).strip().lower()

                # VERY IMPORTANT:
                # Gemini cannot send symptoms
                # outside CareAI's feature list.
                if (
                    symptom
                    in self.feature_set

                    and symptom
                    not in canonical
                ):

                    canonical.append(
                        symptom
                    )

            # BANGLISH_ASE_ACHE_GUARD
            # Banglish "ase" means "আছে", not English "ache".

            message_words = set(
                message.lower().split()
            )

            banglish_has_words = {
                "ase",
                "asey",
                "achey",
            }

            pain_markers = {
                "pain",
                "betha",
                "byatha",
                "jontrona",
                "aching",
                "headache",
                "backache",
            }

            has_banglish_ase = bool(
                message_words
                & banglish_has_words
            )

            has_pain_context = any(
                marker in message.lower()
                for marker in pain_markers
            )

            if (
                "ache" in canonical
                and has_banglish_ase
                and not has_pain_context
            ):
                canonical.remove("ache")

            negated = []

            for symptom in (
                parsed.negated_symptoms
            ):

                symptom = str(
                    symptom
                ).strip().lower()

                if (
                    symptom
                    in self.feature_set

                    and symptom
                    not in negated
                ):

                    negated.append(
                        symptom
                    )

            # Never send explicitly negated
            # symptoms into prediction.
            canonical = [
                symptom

                for symptom
                in canonical

                if symptom
                not in negated
            ]

            return {
                "language":
                    parsed.language,

                "intent":
                    parsed.intent,

                "intent_confidence":
                    float(
                        parsed.intent_confidence
                    ),

                "model_input":
                    canonical,

                "negated_symptoms":
                    negated,

                "clarification_needed":
                    bool(
                        parsed
                        .clarification_needed
                    ),

                "follow_up_question":
                    parsed
                    .follow_up_question,

                "gemini_used":
                    True,
            }

        except Exception as exc:

            print(
                "Gemini symptom NLP failed:",
                exc,
            )

            # Existing CareAI normalizer
            # will continue as fallback.
            return None
