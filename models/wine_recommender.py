"""
Wine Recommender Model
Handles wine recommendation logic
"""

def get_wine_recommendations(text_input):
    """
    Generate wine recommendations based on user input
    
    Args:
        text_input (str): User's input text for wine preferences
        
    Returns:
        list: List of wine dictionaries with rank, name, description, country, and price
    """
    # TODO: Implement actual recommendation algorithm here
    # For now, returning sample data
    
    wines = [
        {
            "rank": 1,
            "name": "Château Margaux 2015",
            "description": "Full-bodied red wine with notes of blackcurrant, cedar, and vanilla. Elegant tannins with a long finish.",
            "country": "France",
            "price": "$850"
        },
        {
            "rank": 2,
            "name": "Cloudy Bay Sauvignon Blanc",
            "description": "Crisp white wine featuring tropical fruit flavors with hints of passionfruit and citrus. Refreshing acidity.",
            "country": "New Zealand",
            "price": "$35"
        },
        {
            "rank": 3,
            "name": "Dom Pérignon Vintage",
            "description": "Premium champagne with complex aromas of white flowers, citrus, and brioche. Creamy texture with fine bubbles.",
            "country": "France",
            "price": "$220"
        },
        {
            "rank": 4,
            "name": "Penfolds Grange",
            "description": "Iconic Australian Shiraz with rich flavors of dark berries, chocolate, and spice. Powerful and age-worthy.",
            "country": "Australia",
            "price": "$625"
        },
        {
            "rank": 5,
            "name": "Riesling Kabinett",
            "description": "Semi-sweet German white wine with peach and apricot notes. Perfect balance of sweetness and acidity.",
            "country": "Germany",
            "price": "$28"
        },
        {
            "rank": 6,
            "name": "Barolo Reserva",
            "description": "Italian red wine with cherry, rose, and tar aromas. Firm tannins and excellent aging potential.",
            "country": "Italy",
            "price": "$95"
        },
        {
            "rank": 7,
            "name": "Chardonnay Reserve",
            "description": "Oak-aged white wine with butter, vanilla, and ripe apple flavors. Smooth and luxurious mouthfeel.",
            "country": "USA",
            "price": "$42"
        },
    ]
    
    return wines
