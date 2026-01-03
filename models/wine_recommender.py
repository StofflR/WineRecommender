"""
Wine Recommender Model
Handles wine recommendation logic using neural network and TF-IDF
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# Import neural network
from models.neural_network import WineRecommenderNN

# Global variables for loaded data
df_wines = None
feature_vectors = None
vectorizer = None
nn_model = None
wine_embeddings = None

def load_data():
    """Load preprocessed data from models/data directory"""
    global df_wines, feature_vectors, vectorizer, nn_model, wine_embeddings
    
    data_dir = Path(__file__).parent / 'data'
    
    # Load cleaned wine data
    df_wines = pd.read_csv(data_dir / 'cleaned_wine_data.csv')
    print(f"Loaded {len(df_wines)} wines")
    
    # Load feature vectors
    with open(data_dir / 'feature_vectors.pkl', 'rb') as f:
        feature_vectors = pickle.load(f)
    print(f"Loaded feature vectors: {feature_vectors.shape}")
    
    # Load vectorizer
    with open(data_dir / 'vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"Loaded vectorizer with {len(vectorizer.get_feature_names_out())} features")
    
    # Try to load trained neural network
    trained_model_path = Path(__file__).parent / 'trained' / 'wine_nn_model.pkl'
    if trained_model_path.exists():
        nn_model = WineRecommenderNN(
            input_dim=feature_vectors.shape[1],
            hidden_dim=128,
            output_dim=64
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
    
    # Extract potential wine attributes
    cleaned_terms = []
    
    # Common wine varieties (subset - extend as needed)
    varieties = ['chardonnay', 'cabernet', 'merlot', 'pinot', 'noir', 'sauvignon', 
                 'blanc', 'riesling', 'shiraz', 'syrah', 'malbec', 'zinfandel']
    
    # Countries
    countries = ['france', 'italy', 'spain', 'usa', 'america', 'american', 
                 'australia', 'argentina', 'chile', 'germany', 'portugal', 'new zealand']
    
    # Price categories
    price_terms = ['budget', 'cheap', 'affordable', 'expensive', 'premium', 'luxury']
    
    # Add prefixes for structured features
    for variety in varieties:
        if variety in text_lower:
            cleaned_terms.append(f"variety_{variety}")
    
    for country in countries:
        if country in text_lower:
            cleaned_terms.append(f"country_{country}")
    
    for price_term in price_terms:
        if price_term in text_lower:
            if price_term in ['budget', 'cheap', 'affordable']:
                cleaned_terms.append("pricecat_budget")
            elif price_term in ['expensive', 'premium', 'luxury']:
                cleaned_terms.append("pricecat_premium")
            else:
                cleaned_terms.append("pricecat_mid_range")
    
    return ' '.join(cleaned_terms) if cleaned_terms else text_lower

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
    
    # Define flavor keywords (same as in training)
    flavor_keywords = {
        'fruit': ['berry', 'cherry', 'apple', 'citrus', 'tropical', 'fruit', 'blackberry', 'raspberry'],
        'dry': ['dry', 'crisp', 'tannic'],
        'sweet': ['sweet', 'honey', 'ripe', 'jam'],
        'oak': ['oak', 'vanilla', 'toast', 'cedar'],
        'spice': ['spice', 'pepper', 'cinnamon', 'clove'],
        'herbal': ['herbal', 'grass', 'mineral', 'earth']
    }
    
    # Check flavor type in description
    for flavor_type, keywords in flavor_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            flavor_terms.append(flavor_type)
    
    return ' '.join(flavor_terms)

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
        
        recommendations.append({
            "rank": rank,
            "name": wine['title'] if 'title' in wine else wine.get('variety', 'Unknown'),
            "description": wine.get('description', 'No description available')[:200] + '...',
            "country": wine.get('country', 'Unknown'),
            "price": f"${int(wine['price'])}" if pd.notna(wine.get('price')) else 'N/A'
        })
    
    return recommendations

def main():
    """Main function to train and save the neural network model"""
    print("Loading data...")
    load_data()
    
    print("\nInitializing neural network...")
    model = WineRecommenderNN(
        input_dim=feature_vectors.shape[1],
        hidden_dim=128,
        output_dim=64
    )
    
    print("\nTraining neural network...")
    model.train_model(feature_vectors, epochs=50, batch_size=128, learning_rate=0.001, verbose=True)
    
    # Save trained model
    trained_dir = Path(__file__).parent / 'trained'
    model.save_model(str(trained_dir / 'wine_nn_model.pkl'))
    
    print("\nTraining complete! Model saved to models/trained/wine_nn_model.pkl")
    
    # Test recommendations
    print("\n" + "="*50)
    print("Testing recommendations...")
    print("="*50)
    
    test_queries = [
        "I like fruity red wines from France",
        "Looking for a crisp white wine",
        "Premium Italian wine with oak flavors"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        recs = get_wine_recommendations(query, top_n=3)
        for rec in recs:
            print(f"  {rec['rank']}. {rec['name']} - {rec['country']} ({rec['price']})")

if __name__ == "__main__":
    main()
