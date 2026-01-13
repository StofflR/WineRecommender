from flask import Flask, render_template, request, jsonify
import time
import argparse
from models.wine_recommender import get_wine_recommendations

app = Flask(__name__)

# Global variables for configuration
device_preference = "cuda"
use_neural_network = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    """Callback function that processes form data and returns wine recommendations"""
    # Simulate processing delay
    time.sleep(2)

    text_input = request.form.get("text_input", "")

    # Get wine recommendations from the model
    wines = get_wine_recommendations(
        text_input, device=device_preference, use_nn=use_neural_network
    )

    return jsonify({"wines": wines})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine Recommender Flask App")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for computations instead of GPU",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask app in debug mode",
    )
    parser.add_argument(
        "--no-nn",
        action="store_true",
        help="Disable neural network and use TF-IDF similarity only",
    )

    args = parser.parse_args()
    device_preference = "cpu" if args.cpu else "cuda"
    use_neural_network = not args.no_nn

    mode = "TF-IDF only" if args.no_nn else "Neural Network"
    print(
        f"Starting Wine Recommender App with device: {device_preference}, mode: {mode}"
    )
    app.run(debug=args.debug)
