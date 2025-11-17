"""Smart Summarizer Flask application."""
from typing import Optional

from flask import Flask, render_template, request
from openai import OpenAI, OpenAIError

# Create the Flask application instance and OpenAI client.
app = Flask(__name__)
client = OpenAI()

# Define constants for summarization behavior.
SYSTEM_PROMPT = "You are a helpful assistant that summarizes text clearly and concisely."
SUMMARY_OPTIONS = {
    "short": "Short Summary (1–2 sentences)",
    "detailed": "Detailed Summary",
    "bullet": "Bullet Point Summary",
    "kid": "Kid-Friendly Summary",
}
LANGUAGE_OPTIONS = ["English", "Spanish", "Hindi", "Telugu", "French"]


def _extract_text_output(response) -> Optional[str]:
    """Safely extract textual output from an OpenAI Responses API result."""

    # Prefer the helper attribute if present (available when response_format defaults to text).
    if getattr(response, "output_text", None):
        return response.output_text.strip()

    output = getattr(response, "output", None)
    if output:
        first_item = output[0]
        content = getattr(first_item, "content", None)
        if content:
            text_piece = getattr(content[0], "text", None)
            if text_piece:
                return text_piece.strip()
    return None


def generate_summary(text: str, summary_type: str, target_language: str) -> Optional[str]:
    """Generate a summary using the OpenAI Responses API."""

    style_instructions = {
        "short": "Provide a brief 1–2 sentence summary.",
        "detailed": "Provide a detailed, multi-paragraph summary as needed.",
        "bullet": "Provide a concise bullet list summary using clear bullet points.",
        "kid": "Explain the summary in simple, kid-friendly language.",
    }

    summary_style = style_instructions.get(summary_type, style_instructions["short"])
    translation = f"Output the summary in {target_language}."

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Summarize the following text. {summary_style} {translation} "
                            f"Text:\n{text}"
                        ),
                    }
                ],
            },
        ],
    )

    return _extract_text_output(response)


def analyze_sentiment(text: str) -> Optional[str]:
    """Analyze the sentiment (positive/neutral/negative) of the given text."""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "You are a concise sentiment analyst that replies with Positive, Neutral, or Negative.",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Determine sentiment of:\n{text}"}],
            },
        ],
    )

    return _extract_text_output(response)


@app.route("/")
def index():
    """Render the home page with the summarization form."""
    return render_template(
        "index.html",
        summary=None,
        sentiment=None,
        selected_summary="short",
        selected_language="English",
        sentiment_enabled=False,
        original_text="",
    )


@app.route("/summarize", methods=["POST"])
def summarize():
    """Return an AI-generated summary for the submitted text with optional sentiment."""

    text = request.form.get("text", "").strip()
    summary_type = request.form.get("summary_type", "short")
    target_language = request.form.get("target_language", "English")
    sentiment_enabled = request.form.get("sentiment", "off") == "on"

    summary: Optional[str]
    sentiment: Optional[str] = None

    if not text:
        summary = "Error: Please provide text to summarize."
    else:
        try:
            summary = generate_summary(text, summary_type, target_language)
            if not summary:
                summary = "Error: Unable to generate a summary."
        except OpenAIError as exc:
            summary = f"Error: Failed to generate summary ({exc})."
        else:
            if sentiment_enabled:
                try:
                    sentiment = analyze_sentiment(text)
                except OpenAIError as exc:
                    sentiment = f"Error: Failed to analyze sentiment ({exc})."

    return render_template(
        "index.html",
        summary=summary,
        sentiment=sentiment,
        selected_summary=summary_type,
        selected_language=target_language,
        sentiment_enabled=sentiment_enabled,
        original_text=text,
    )


if __name__ == "__main__":
    # Run the Flask development server for local testing.
    app.run(debug=True)
