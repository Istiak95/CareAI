from __future__ import annotations

import os
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

NEGATION:

Understand negation in English, Bangla,
and Banglish.

Example:

"amar jor nai but kashi ase"

If the allowed list contains fever and cough:

fever = negated symptom
cough = present symptom

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
