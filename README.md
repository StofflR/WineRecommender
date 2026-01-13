# Wine Recommender

An intelligent wine recommendation system that leverages deep learning to translate natural language wine preferences into relevant recommendations. The system uses a symmetric autoencoder neural network to learn compressed 64-dimensional embeddings from wine descriptions, enabling semantic similarity-based retrieval from a database of 129,970 wines.

## Project Overview

This project implements a complete wine recommendation pipeline consisting of:
- Data preprocessing with TF-IDF vectorization (1500 features)
- Symmetric autoencoder architecture (1500→512→384→256→64→256→384→512→1500)
- Cosine similarity-based recommendation generation
- Flask web application with interactive query interface
- Evaluation framework using MRR, MAP, and nDCG metrics

The system compares the deep learning approach against a TF-IDF baseline to demonstrate the effectiveness of learned semantic embeddings.

## Dataset

The project uses the Wine Reviews dataset by Zackthoutt, available on Kaggle:
- Source: https://www.kaggle.com/datasets/zynicide/wine-reviews
- Size: 129,970 wine reviews
- Location: After preprocessing, the cleaned data is stored in `models/data/cleaned_wine_data.csv`
- Features: Wine descriptions, variety, country, price, points, and other metadata

Download the dataset from Kaggle and place it in the appropriate directory before running data processing scripts.

## Project Structure

```
WineRecommender/
├── app.py                         # Flask web application
├── requirements.txt               # Python dependencies
├── data_processing/
│   └── dataprocessing.ipynb       # Data cleaning and preprocessing
├── models/
│   ├── neural_network.py          # Autoencoder implementation
│   ├── wine_recommender.py        # Recommendation engine
│   ├── keywords.py                # Keyword extraction utilities
│   ├── data/
│   │   └── cleaned_wine_data.csv  # Processed dataset
│   ├── trained/                   # Saved model weights
│   │   └── 512-384-256-64/        # Best performing architecture
│   └── plots/                     # Training visualizations
├── evaluation/
│   ├── evaluation.ipynb           # Evaluation metrics computation
│   ├── scoring.ipynb              # Manual scoring interface
│   ├── queries.txt                # Test queries
│   └── scores*.json               # Evaluation results
└── templates/
    └── index.html                 # Main web interface
```

## Experimental Setup

### 1. Data Preprocessing

Process the raw Kaggle dataset:
```bash
jupyter notebook data_processing/dataprocessing.ipynb
```

This notebook:
- Cleans and filters wine reviews
- Extracts relevant features
- Generates TF-IDF vectors (1500 dimensions)
- Saves processed data to `models/data/cleaned_wine_data.csv`

### 2. Model Training

Train the autoencoder neural network:
```bash
python models/neural_network.py
```

Training configuration:
- Architecture: 1500→512→384→256→64→256→384→512→1500
- Optimizer: Adam (learning rate 0.0005, weight decay 1e-5)
- Loss: Mean Squared Error (MSE)
- Batch size: 256
- Epochs: 250
- Regularization: Dropout (0.3, 0.2, 0.2) and Batch Normalization

Trained model weights are saved to `models/trained/512-384-256-64/`.

### 3. Running the Web Application

Start the Flask web server:
```bash
python app.py [OPTIONS]
```

Available options:
- `--cpu` - Run Flask app using CPU only
- `--debug` - Run Flask app in debug mode

Examples:
```bash
python app.py                    # Run with CUDA (if available) 
python app.py --cpu              # Run with CPU only
python app.py --debug            # Run with CUDA in debug mode
python app.py --cpu --debug      # Run with CPU in debug mode
```

The application will be available at `http://localhost:5000`

### 4. Scoring and Evaluation


Run the scoring interface to generate recommendations:
```bash
jupyter notebook evaluation/scoring.ipynb
```

Compute metrics:
```bash
jupyter notebook evaluation/evaluation.ipynb
```
