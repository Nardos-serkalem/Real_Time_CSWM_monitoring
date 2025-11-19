from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Path to your plots folder
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "collectors", "assets", "plots")

@app.route("/")
def home():
    return "Flask backend is running. Use /plots to see available plots."


@app.route("/plots/<filename>")
def get_plot(filename):
    """Serve a specific plot file."""
    if os.path.exists(os.path.join(PLOTS_DIR, filename)):
        return send_from_directory(PLOTS_DIR, filename)
    else:
        return jsonify({"error": "File not found"}), 404

@app.route("/plots")
def list_plots():
    """Return a list of all plot files."""
    files = [f for f in os.listdir(PLOTS_DIR) if os.path.isfile(os.path.join(PLOTS_DIR, f))]
    return jsonify(files)

if __name__ == "__main__":
    logging.info("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
