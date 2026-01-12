"""
Wine Recommender Model
Handles wine recommendation logic using neural network and TF-IDF
"""

import re
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Import neural network
from models.neural_network import WineRecommenderNN

# Import keywords from single source of truth
from models.keywords import country_keywords, price_keywords, flavor_keywords

# Global variables for loaded data
df_wines = None
feature_vectors = None
vectorizer = None
nn_model = None
wine_embeddings = None

# Global keyword dictionaries (initialized in load_data)
unique_varieties = []
normalized_unique_varieties = []
unique_regions = []
normalized_unique_regions = []
unique_wineries = []
normalized_unique_wineries = []


def load_data(hidden_dim=512, output_dim=64):
    """Load preprocessed data from models/data directory"""
    global df_wines, feature_vectors, vectorizer, nn_model, wine_embeddings
    global unique_varieties, normalized_unique_varieties
    global unique_regions, normalized_unique_regions
    global unique_wineries, normalized_unique_wineries

    data_dir = Path(__file__).parent / "data"

    # Load cleaned wine data
    df_wines = pd.read_csv(data_dir / "cleaned_wine_data.csv")
    print(f"Loaded {len(df_wines)} wines")

    # Extract unique values from dataset
    unique_varieties = df_wines["variety"].unique().tolist()
    normalized_unique_varieties = [
        re.sub(r"[ -]", "_", variety) for variety in unique_varieties
    ]

    unique_regions = df_wines["region_1"].unique().tolist()
    normalized_unique_regions = [
        re.sub(r"[ -]", "_", region) for region in unique_regions
    ]

    unique_wineries = df_wines["winery"].unique().tolist()
    normalized_unique_wineries = [
        re.sub(r"[ -]", "_", winery) for winery in unique_wineries
    ]

    # Load feature vectors
    with open(data_dir / "feature_vectors.pkl", "rb") as f:
        feature_vectors = pickle.load(f)
    print(f"Loaded feature vectors: {feature_vectors.shape}")

    # Load vectorizer
    with open(data_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print(f"Loaded vectorizer with {len(vectorizer.get_feature_names_out())} features")

    # Try to load trained neural network
    trained_model_path = Path(__file__).parent / "trained" / "wine_nn_model.pt"
    print(f"Loading trained model from {trained_model_path}")
    if trained_model_path.exists():
        nn_model = WineRecommenderNN(
            input_dim=feature_vectors.shape[1],
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        nn_model.load_model(str(trained_model_path))

        # Generate embeddings for all wines
        wine_embeddings = nn_model.get_embeddings(feature_vectors)
        print(f"Generated wine embeddings: {wine_embeddings.shape}")
    else:
        print("Warning: Trained model not found. Using TF-IDF similarity only.")
        nn_model = None
        wine_embeddings = None


def get_user_input_tfidf(text_input):
    """
    Convert user text input to TF-IDF vector using the trained vectorizer.
    Only includes terms that were learned during training.

    Args:
        text_input (str): User's input text for wine preferences

    Returns:
        sparse matrix: TF-IDF vector for user input
    """
    if vectorizer is None:
        raise RuntimeError("Data not loaded. Call load_data() first.")

    # Clean and structure the input similar to training data
    cleaned_input = clean_user_input(text_input)

    # Extract flavor features from input
    flavor_profile = extract_flavor_features(text_input)

    # Create structured features similar to training
    structured_input = f"{cleaned_input} flavor_{flavor_profile}"

    # Transform using the trained vectorizer
    # This automatically filters to only learned terms
    user_vector = vectorizer.transform([structured_input])

    return user_vector


def clean_user_input(text_input):
    """
    Clean user input to match training data format

    Args:
        text_input (str): Raw user input

    Returns:
        str: Cleaned input string
    """
    if not text_input or not isinstance(text_input, str):
        return ""

    text_lower = text_input.lower().strip()

    # Split into words for multi-word matching
    words = text_lower.split()

    # Generate word combinations (single, two-word, three-word phrases)
    two_words = [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]
    three_words = [
        words[i] + "_" + words[i + 1] + "_" + words[i + 2]
        for i in range(len(words) - 2)
    ]

    # Extract potential wine attributes
    cleaned_terms = []

    for country, value in country_keywords.items():
        for keyword in value:
            for word in words:
                if word == keyword:
                    cleaned_terms.append("country_" + country)

    for flavor, value in flavor_keywords.items():
        for keyword in value:
            for word in words:
                if word == keyword:
                    cleaned_terms.append("flavor_" + flavor)

    for price, value in price_keywords.items():
        for keyword in value:
            for word in words:
                if word == keyword:
                    cleaned_terms.append("pricecat_" + price)
            for pair in two_words:
                if pair == keyword:
                    cleaned_terms.append("pricecat_" + price)

    for variety in normalized_unique_varieties:
        for word in words:
            if word == variety.lower():
                cleaned_terms.append("variety_" + word)
        for pair in two_words:
            if pair == variety.lower():
                cleaned_terms.append("variety_" + pair)
        for triple in three_words:
            if triple == variety.lower():
                cleaned_terms.append("variety_" + triple)

    for region in normalized_unique_regions:
        for word in words:
            if word == region.lower():
                cleaned_terms.append("region_" + word)
        for pair in two_words:
            if pair == region.lower():
                cleaned_terms.append("region_" + pair)
        for triple in three_words:
            if triple == region.lower():
                cleaned_terms.append("region_" + triple)

    for winery in normalized_unique_wineries:
        for word in words:
            if word == winery.lower():
                cleaned_terms.append("winery_" + word)
        for pair in two_words:
            if pair == winery.lower():
                cleaned_terms.append("winery_" + pair)
        for triple in three_words:
            if triple == winery.lower():
                cleaned_terms.append("winery_" + triple)

    return " ".join(cleaned_terms) if cleaned_terms else text_lower


def extract_flavor_features(text_input):
    """
    Extract flavor profile from user input (same as training)

    Args:
        text_input (str): User input text

    Returns:
        str: Extracted flavor terms
    """
    if not text_input or not isinstance(text_input, str):
        return ""

    text_lower = text_input.lower()
    flavor_terms = []

    # Check flavor type in description
    for flavor_type, keywords in flavor_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            flavor_terms.append(flavor_type)

    return " ".join(flavor_terms)


def get_wine_recommendations(text_input, top_n=7):
    """
    Generate wine recommendations based on user input

    Args:
        text_input (str): User's input text for wine preferences
        top_n (int): Number of recommendations to return

    Returns:
        list: List of wine dictionaries with rank, name, description, country, and price
    """
    if df_wines is None or feature_vectors is None:
        load_data()

    # Get user input vector
    user_vector = get_user_input_tfidf(text_input)

    # Calculate similarities
    if nn_model is not None and wine_embeddings is not None:
        # Use neural network embeddings
        user_embedding = nn_model.get_embeddings(user_vector)
        similarities = cosine_similarity(user_embedding, wine_embeddings)[0]
    else:
        # Fallback to TF-IDF cosine similarity
        similarities = cosine_similarity(user_vector, feature_vectors)[0]

    # Get top N most similar wines
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # Format results
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        wine = df_wines.iloc[idx]

        recommendations.append(
            {
                "rank": rank,
                "name": (
                    wine["title"] if "title" in wine else wine.get("variety", "Unknown")
                ),
                "description": wine.get("description", "No description available"),
                "country": wine.get("country", "Unknown"),
                "region": wine.get("region_1", wine.get("region_2", "Unknown")),
                "variety": wine.get("variety", "Unknown"),
                "price": (
                    f"${int(wine['price'])}" if pd.notna(wine.get("price")) else "N/A"
                ),
            }
        )

    return recommendations
