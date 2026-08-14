from __future__ import annotations

import json
import re

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
        Explain the existing CareAI result only.
        No new disease suggestion is generated here.
        """

        if not isinstance(previous_result, dict):
            return None

        predictions = (
            previous_result.get("top_predictions")
            or []
        )

        if not predictions:
            return None

        top = predictions[0]

        disease = (
            top.get("disease")
            or "Unknown condition"
        )

        confidence = top.get(
            "confidence_percent"
        )

        if confidence is None:
            try:
                confidence = round(
                    float(
                        top.get("confidence", 0)
                    ) * 100,
                    2,
                )
            except Exception:
                confidence = 0

        symptoms = (
            previous_result.get(
                "matched_symptoms"
            )
            or previous_result.get(
                "extracted_symptoms"
            )
            or []
        )

        # ----------------------------------------------------
        # SHAP: strongest present symptoms only
        # ----------------------------------------------------

        shap_data = (
            top.get("shap_explanation")
            or {}
        )

        shap_items = (
            shap_data.get(
                "present_symptom_contributions"
            )
            or []
        )

        def shap_strength(item):
            try:
                return abs(
                    float(
                        item.get(
                            "contribution",
                            item.get(
                                "shap_value",
                                0,
                            ),
                        )
                    )
                )
            except Exception:
                return 0.0

        shap_items = sorted(
            shap_items,
            key=shap_strength,
            reverse=True,
        )[:3]

        shap_symptoms = [
            str(item.get("symptom"))
            for item in shap_items
            if item.get("symptom")
        ]

        # ----------------------------------------------------
        # CAREAI EXISTING RECOMMENDATION ONLY
        # ----------------------------------------------------

        recommendation = (
            top.get("recommendation")
            or {}
        )

        doctor = (
            recommendation.get(
                "doctor_type_patient_should_see"
            )
            or recommendation.get(
                "doctor_type"
            )
            or recommendation.get(
                "specialist"
            )
            or recommendation.get(
                "see"
            )
        )

        tests = (
            recommendation.get(
                "common_tests_to_discuss_with_clinician"
            )
            or recommendation.get(
                "tests"
            )
            or []
        )

        if isinstance(tests, str):
            tests = [
                x.strip()
                for x in tests
                .replace(";", ",")
                .split(",")
                if x.strip()
            ]

        tests = list(tests)[:5]

        # ----------------------------------------------------
        # LANGUAGE FROM ACTUAL USER REQUEST
        # ----------------------------------------------------

        msg = str(user_message or "").lower()

        has_bangla_script = any(
            "\u0980" <= ch <= "\u09ff"
            for ch in str(user_message or "")
        )

        english_markers = [
            "explain",
            "i can't understand",
            "i cant understand",
            "i cannot understand",
            "i don't understand",
            "i dont understand",
            "what does this mean",
        ]

        banglish_markers = [
            "bujai",
            "bujhai",
            "bujhi na",
            "bujhi nai",
            "bujhlam na",
            "bujlam na",
            "sohoj kore",
            "easy kore",
            "amk ",
            "amake ",
        ]

        if has_bangla_script:
            output_language = "bangla"

        elif any(
            marker in msg
            for marker in english_markers
        ):
            output_language = "english"

        elif any(
            marker in msg
            for marker in banglish_markers
        ):
            output_language = "banglish"

        elif language in {
            "bangla",
            "banglish",
            "english",
        }:
            output_language = language

        else:
            output_language = "english"

        data = {
            "condition": disease,
            "confidence_percent": confidence,
            "present_symptoms": symptoms,
            "shap_symptoms": shap_symptoms,
            "doctor_suggestion": doctor,
            "test_suggestions": tests,
        }

        if output_language == "bangla":

            language_rule = """
Respond in natural Bangla script.

Keep disease names, symptom names, doctor types,
test names, SHAP and percentages in English when useful.

Use natural wording similar to:

সহজভাবে বললে, CareAI তোমার symptom-এর ভিত্তিতে
X-কে সবচেয়ে সম্ভাব্য condition হিসেবে suggest করেছে,
confidence Y%। SHAP অনুযায়ী ... এই suggestion-এ
সবচেয়ে বেশি প্রভাব ফেলেছে। এই result অনুযায়ী ...
doctor-এর সঙ্গে কথা বলা যেতে পারে এবং প্রয়োজন হলে
... test নিয়ে doctor-এর সঙ্গে আলোচনা করা যেতে পারে।
তবে এটি নিশ্চিত diagnosis নয়।
"""

        elif output_language == "banglish":

            language_rule = """
Respond in natural Banglish using Latin letters only.

Do not use Bangla script.

Use natural wording similar to:

Sohoj vabe bolle, CareAI tomar symptom-er upor base kore
X-ke shobcheye probable condition hisebe suggest koreche,
confidence Y%. SHAP onujayi ... ei suggestion-e shobcheye
beshi influence koreche. Ei result onujayi ... doctor-er
sathe kotha bola jete pare, ebong proyojon hole ... test
niye doctor-er sathe alochona kora jete pare.
Tobe eta confirmed diagnosis noy.
"""

        else:

            language_rule = """
Respond in simple natural English.

Use natural wording similar to:

In simple terms, CareAI suggests X as the most likely
condition based on the detected symptoms, with Y%
confidence. According to SHAP, ... had the strongest
influence on this suggestion. Based on this result,
the user may consider speaking with ... and, if needed,
discussing ... tests with their doctor.
This is not a confirmed diagnosis.
"""

        prompt = f"""
You are CareAI's explanation assistant.

Explain ONLY the existing CareAI result below.

{language_rule}

STRICT RULES:

- Keep the complete explanation to 3-5 short sentences.
- Say "CareAI", not "the model".
- Use "suggest" or "suggestion", not "prediction".
- Mention only the TOP-1 condition.
- Mention the confidence.
- Mention the detected present symptoms naturally.
- Mention at most 2-3 SHAP symptom influences.
- If a doctor suggestion exists, mention it naturally.
- If test suggestions exist, mention them naturally.
- Use ONLY doctor/test information supplied by CareAI.
- Do NOT invent doctors or tests.
- Do NOT add another disease.
- Do NOT add treatment or medicine advice.
- Do NOT add new medical facts.
- Do NOT add new emergency advice.
- Do NOT use headings.
- Do NOT use bullet points.
- End with one short sentence saying this is not a
  confirmed diagnosis.

USER REQUEST:
{user_message}

CAREAI DATA:
{json.dumps(data, ensure_ascii=False)}
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

            answer = str(
                interaction.output_text
                or ""
            ).strip()

            return answer or None

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

IMPORTANT FOLLOW-UP CONTEXT:

The application may send a message like:

Original vague symptom text
User clarification: more specific answer

When "User clarification:" exists:

- intent MUST be symptom_followup
- use BOTH parts as context
- the clarification resolves the ambiguity
- do NOT preserve incompatible guesses from the vague text
- return only canonical symptoms supported by the resolved meaning
- do NOT invent extra body locations

Example:

amar betha hoche
User clarification: buker ba pashe

This means chest pain.

Return:
canonical_symptoms = ["pain chest"]

Do NOT return:
pain foot
back pain
abdominal pain
or another unsupported pain location.

This rule is generic. The clarification answer may resolve
location, symptom type, severity, presence/absence, or other
missing information.

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

Ask ONE short clarification question when the user
appears to describe a health symptom but mapping the
message to a canonical symptom would require guessing.

Examples of missing information can include:

- body location
- symptom type
- what the user actually means
- whether a symptom is present or absent
- another detail required to distinguish between
  possible canonical symptoms

IMPORTANT:

Do NOT invent:
- a body location
- a symptom
- severity
- duration
- presence or absence
- a specific canonical feature

If the input is vague, prefer clarification over guessing.

When clarification is required:

clarification_needed = true

Generate exactly ONE short follow_up_question.

Use the same language style as the user:

Banglish user -> Banglish question
Bangla user -> Bangla question
English user -> English question

Examples:

User:
"amar betha hocche"

Do NOT guess:
pain foot
pain chest
back pain
abdominal pain
or another body location.

Return no guessed location-specific symptom.

Ask:
"Betha ta kothay hocche?"

User:
"I have pain"

Ask:
"Where are you feeling the pain?"

User:
"আমার ব্যথা হচ্ছে"

Ask:
"ব্যথাটা কোথায় হচ্ছে?"

User:
"amar kichu problem hocche"

If the actual symptom cannot be identified,
ask what problem or symptom the user is experiencing.

User:
"amar shash e problem"

If it is unclear whether this means shortness of breath
or something else, ask one brief clarification question.

If the symptom is already clear enough to map safely,
do NOT ask an unnecessary clarification question.

If the text is unrelated, meaningless, greeting, or
an explanation request, follow the intent rules above
instead of asking a medical clarification question.

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
