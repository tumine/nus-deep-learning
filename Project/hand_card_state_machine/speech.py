from intent_parser import parse_request
from flask import Flask, render_template, request
import os
import whisper

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = whisper.load_model("base")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_audio", methods=["POST"])
def upload_audio():

    if "audio" not in request.files:
        return {"status": "error"}

    audio_file = request.files["audio"]

    save_path = os.path.join(
        UPLOAD_FOLDER,
        "student_audio.webm"
    )

    audio_file.save(save_path)

    print(f"[INFO] Audio saved: {save_path}")

    result = model.transcribe(save_path)

    text = result["text"]

    request_message = parse_request(text)

    print("Speech:", text)
    print("Request:", request_message)

    return {
        "status": "success",
        "text": text,
        "request": request_message
    }

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )
