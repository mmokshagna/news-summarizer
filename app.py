"""Smart Summarizer Flask application."""
from typing import Optional

from flask import Flask, render_template, request
from openai import OpenAI, OpenAIError

# Create the Flask application instance and OpenAI client.
app = Flask(__name__)
client = OpenAI()

# Define the reusable system prompt for summarization.
SYSTEM_PROMPT = "You are a helpful assistant that summarizes text clearly and concisely."


def generate_summary(text: str) -> Optional[str]:
    """Generate a summary of ``text`` using OpenAI's Chat Completions API."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.7,
    )
    message = response.choices[0].message
    return message.content.strip() if message and message.content else None


@app.route("/")
def index():
    """Render the home page with the summarization form."""
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    """Return an AI-generated summary for the submitted text."""
    text = request.form.get("text", "").strip()

    if not text:
        summary = "Error: Please provide text to summarize."
    else:
        try:
            summary = generate_summary(text) or "Error: Unable to generate a summary."
        except OpenAIError as exc:
            summary = f"Error: Failed to generate summary ({exc})."

    return render_template("index.html", summary=summary)


if __name__ == "__main__":
    # Run the Flask development server for local testing.
    app.run(debug=True)
