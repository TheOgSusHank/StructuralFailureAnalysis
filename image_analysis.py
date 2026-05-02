"""
Crack image analysis via OpenAI Vision (gpt-5).
Uses Replit AI Integrations — no separate API key required.
"""
from __future__ import annotations

import base64
import json
import os
import re

from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
MODEL = "gpt-5"

SYSTEM_PROMPT = """You are a structural engineering AI assistant specialised in
fatigue crack analysis. When given a photograph you must:
1. Determine whether a crack is visible in the image.
2. If a crack is found, characterise it as precisely as possible.
3. Return ONLY a valid JSON object — no markdown fences, no explanation text.

Return exactly this schema (all fields required):
{
  "crack_detected": true | false,
  "crack_type": string,          // e.g. "hairline", "fatigue", "corrosion", "structural", "surface", "none"
  "crack_length_estimate": string, // e.g. "~12 mm (estimated from context)" or "unknown — no scale reference"
  "crack_width_estimate": string,
  "orientation": string,         // e.g. "diagonal", "transverse", "longitudinal", "branching", "none"
  "surface_condition": string,   // e.g. "corroded", "clean", "painted", "weathered"
  "severity": "low" | "moderate" | "high" | "critical",
  "confidence": "low" | "medium" | "high",
  "numeric_estimates": {
    "crack_length_mm": number | null,   // null if scale cannot be determined
    "stress_intensity": number | null,  // MPa·√m estimate, null if indeterminate
    "load_cycles": number | null        // null — image alone cannot determine this
  },
  "findings": string,            // 2-4 sentence plain-English summary of what was found
  "recommended_action": string   // specific engineering recommendation
}"""

USER_PROMPT_TEMPLATE = """Analyse this photograph for structural cracks.
{scale_hint}
Respond with ONLY the JSON object described in your system instructions."""


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
        base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
    )


def analyse_crack_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    scale_reference: str | None = None,
) -> dict:
    """
    Send an image to GPT-5 Vision and return structured crack analysis.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, WEBP).
        mime_type: MIME type of the image.
        scale_reference: Optional text giving scale context,
                         e.g. "The ruler in the photo shows 10 cm total width."

    Returns:
        Parsed JSON dict matching the schema above.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type}; base64,{b64}"

    scale_hint = (
        f"Scale reference provided by user: {scale_reference}"
        if scale_reference
        else "No scale reference was provided — use visual context to estimate where possible."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT_TEMPLATE.format(scale_hint=scale_hint),
                    },
                ],
            },
        ],
        max_completion_tokens=1024,
    )

    raw = response.choices[0].message.content or "{}"

    # Strip any accidental markdown fences the model might emit
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return a structured error so the UI can handle it cleanly
        return {
            "crack_detected": False,
            "crack_type": "parse_error",
            "crack_length_estimate": "unknown",
            "crack_width_estimate": "unknown",
            "orientation": "unknown",
            "surface_condition": "unknown",
            "severity": "low",
            "confidence": "low",
            "numeric_estimates": {
                "crack_length_mm": None,
                "stress_intensity": None,
                "load_cycles": None,
            },
            "findings": "The AI response could not be parsed. Please try again.",
            "recommended_action": "Re-upload the image and retry the analysis.",
            "_raw": raw,
        }
