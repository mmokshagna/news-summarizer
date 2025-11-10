"""Smart Summarizer Flask application."""
from flask import Flask, render_template, request

# Create the Flask application instance.
app = Flask(__name__)


@app.route("/")
def index():
    """Render the home page with the summarization form."""
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    """Return a dummy summary for the submitted text."""
    # Retrieve the text input from the form submission.
    _ = request.form.get("text", "")

    # In a real application, you would process `_` to generate a summary.
    summary = "This is your summary."

    # Render the template with the summary message.
    return render_template("index.html", summary=summary)


if __name__ == "__main__":
    # Run the Flask development server for local testing.
    app.run(debug=True)
