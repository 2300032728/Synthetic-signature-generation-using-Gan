# api.py
import os
import io
import zipfile
import base64
from flask import Flask, request, jsonify, send_file
from generate_signatures import generate_signatures

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    n = data.get("n", 1)
    user_id = data.get("user_id", "")

    output_dir = "generated_signatures_api"
    os.makedirs(output_dir, exist_ok=True)

    # Generate signatures
    image_paths = generate_signatures(N=n, output_dir=output_dir, user_id=user_id)

    # Create in-memory ZIP
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zipf:
        for img_path in image_paths:
            zipf.write(img_path, os.path.basename(img_path))
    memory_file.seek(0)

    return send_file(memory_file, attachment_filename="signatures.zip", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
