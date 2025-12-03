from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("✅ Summarization model loaded.")
except Exception as e:
    print("❌ Error loading summarization model:", e)
    summarizer = None


@app.route("/summarize", methods=["POST"])
def summarize():
    """
    Expects JSON:
        { "text": "your long paragraph here..." }
    Returns JSON:
        { "summary": "short summary..." }
    """
    if summarizer is None:
        return jsonify({"error": "Summarization model is not available."}), 500

    input_data = request.get_json(silent=True)
    print("📩 Raw JSON from client:", input_data) 

    if not input_data or "text" not in input_data:
        return jsonify({"error": "Invalid input. Send JSON like {\"text\": \"...\"}"}), 400

    text = input_data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is empty."}), 400

    MAX_CHARS = 8000
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Input text is too long. Limit is {MAX_CHARS} characters."}), 400

    try:
        result = summarizer(
            text,
            max_length=120,  
            min_length=40,  
            do_sample=False
        )
        summary_text = result[0]["summary_text"]
        return jsonify({"summary": summary_text}), 200

    except Exception as e:
        print("❌ Error during summarization:", e)
        return jsonify({"error": "Failed to generate summary."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
