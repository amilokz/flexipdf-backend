from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from utils import pdf_to_word, word_to_pdf, pdf_to_images, images_to_pdf
from chatbot import AliChatbot
from PyPDF2 import PdfReader, PdfWriter
import os
import datetime
import traceback
from mangum import Mangum  # <-- new

app = Flask(__name__)
CORS(app)

# ====== Folder Setup ======
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ====== Initialize Chatbot ======
chatbot = AliChatbot()

def timestamped_filename(filename):
    name, ext = os.path.splitext(filename)
    return f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"

# ====== API ROUTES ======
@app.route('/api/')
def api_home():
    return jsonify({
        "message": "FlexiPDF Backend is running 🚀",
        "routes": [
            "/api/chat",
            "/api/chat/history",
            "/api/chat/clear",
            "/api/convert/pdf-to-word",
            "/api/convert/word-to-pdf",
            "/api/convert/pdf-to-images",
            "/api/convert/images-to-pdf",
            "/api/convert/merge-pdf",
            "/api/convert/split-pdf"
        ]
    })

# ====== Chatbot ======
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        msg = data.get('message', '').strip()
        if not msg:
            return jsonify({"status": "error", "reply": "Empty message"}), 200

        reply = chatbot.get_response(msg)
        return jsonify({"status": "success", "reply": reply}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "reply": str(e)}), 500


@app.route('/api/chat/history', methods=['GET'])
def chat_history():
    return jsonify({"status": "success", "history": chatbot.data.get("conversations", [])})


@app.route('/api/chat/clear', methods=['DELETE'])
def clear_history():
    chatbot.reset_memory()
    return jsonify({"status": "success", "message": "Chat cleared"})


# ====== PDF → WORD ======
@app.route('/api/convert/pdf-to-word', methods=['POST'])
def pdf_to_word_route():
    try:
        file = request.files['file']
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamped_filename(file.filename))
        file.save(in_path)

        out_path = os.path.join(
            app.config['OUTPUT_FOLDER'],
            os.path.basename(in_path).replace('.pdf', '.docx')
        )

        pdf_to_word(in_path, out_path)
        return jsonify({"status": "success", "download_url": f"/api/download/{os.path.basename(out_path)}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== WORD → PDF ======
@app.route('/api/convert/word-to-pdf', methods=['POST'])
def word_to_pdf_route():
    try:
        file = request.files['file']
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamped_filename(file.filename))
        file.save(in_path)

        out_path = os.path.join(
            app.config['OUTPUT_FOLDER'],
            os.path.basename(in_path).replace('.docx', '.pdf')
        )

        word_to_pdf(in_path, out_path)
        return jsonify({"status": "success", "download_url": f"/api/download/{os.path.basename(out_path)}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== PDF → IMAGES ======
@app.route('/api/convert/pdf-to-images', methods=['POST'])
def pdf_to_images_route():
    try:
        file = request.files['file']
        in_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamped_filename(file.filename))
        file.save(in_path)

        images = pdf_to_images(in_path, app.config['OUTPUT_FOLDER'])
        urls = [f"/api/download/{os.path.basename(p)}" for p in images]

        return jsonify({"status": "success", "images": urls})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== IMAGES → PDF ======
@app.route('/api/convert/images-to-pdf', methods=['POST'])
def images_to_pdf_route():
    try:
        files = request.files.getlist('files')
        paths = []

        for f in files:
            p = os.path.join(app.config['UPLOAD_FOLDER'], timestamped_filename(f.filename))
            f.save(p)
            paths.append(p)

        out_name = f"images_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)

        images_to_pdf(paths, out_path)
        return jsonify({"status": "success", "download_url": f"/api/download/{out_name}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== MERGE PDF ======
@app.route('/api/convert/merge-pdf', methods=['POST'])
def merge_pdf():
    try:
        files = request.files.getlist('files')
        writer = PdfWriter()

        for f in files:
            p = os.path.join(app.config['UPLOAD_FOLDER'], timestamped_filename(f.filename))
            f.save(p)
            reader = PdfReader(p)
            for page in reader.pages:
                writer.add_page(page)

        out_name = f"merged_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)

        with open(out_path, "wb") as fp:
            writer.write(fp)

        return jsonify({"status": "success", "download_url": f"/api/download/{out_name}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== SPLIT PDF ======
@app.route('/api/convert/split-pdf', methods=['POST'])
def split_pdf():
    try:
        file = request.files['file']
        p = os.path.join(app.config['UPLOAD_FOLDER'], timestamped_filename(file.filename))
        file.save(p)

        reader = PdfReader(p)
        urls = []

        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)

            name = f"page_{i+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            out = os.path.join(app.config['OUTPUT_FOLDER'], name)

            with open(out, "wb") as fp:
                writer.write(fp)

            urls.append(f"/api/download/{name}")

        return jsonify({"status": "success", "files": urls})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== DOWNLOAD ======
@app.route('/api/download/<filename>')
def download(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"status": "error", "message": "File not found"}), 404


handler = Mangum(app)