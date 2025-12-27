from flask import Flask, render_template, request, jsonify
import time
from models.wine_recommender import get_wine_recommendations

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Callback function that processes form data and returns wine recommendations"""
    # Simulate processing delay
    time.sleep(2)
    
    text_input = request.form.get('text_input', '')
    
    # Get wine recommendations from the model
    wines = get_wine_recommendations(text_input)
    
    return jsonify({'wines': wines})

if __name__ == '__main__':
    app.run(debug=True)
