# backend/main.py
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Path to your plots folder
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "collectors", "assets", "plots")

@app.route("/")
def home():
    return "Flask backend is running. Use /plots and /latest_time."

@app.route("/plots")
def list_plots():
    """Return a list of all plot files."""
    files = [f for f in os.listdir(PLOTS_DIR) if os.path.isfile(os.path.join(PLOTS_DIR, f)) and f.endswith(".png")]
    return jsonify({"plots": files})

@app.route("/plots/<filename>")
def get_plot(filename):
    """Serve a specific plot file."""
    file_path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(PLOTS_DIR, filename)
    else:
        return jsonify({"error": "File not found"}), 404

@app.route("/latest_time")
def latest_time():
    """Return last modified time of the latest plot."""
    plot_files = [os.path.join(PLOTS_DIR, f) for f in os.listdir(PLOTS_DIR) if f.endswith(".png")]
    if not plot_files:
        return jsonify({"latest": None})
    latest_file = max(plot_files, key=os.path.getmtime)
    last_modified = datetime.fromtimestamp(os.path.getmtime(latest_file))
    return jsonify({"latest": last_modified.strftime("%Y-%m-%d %H:%M:%S")})

if __name__ == "__main__":
    logging.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
