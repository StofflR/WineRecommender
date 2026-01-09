"""
Keywords and dictionaries for wine recommendation system
Single source of truth for all keyword mappings
"""

# Country keywords dictionary created with ChatGPT
country_keywords = {
    "italy": ["italy", "italian"],
    "portugal": ["portugal", "portuguese", "portugese"],
    "us": ["us", "usa", "united states", "united states of america"],
    "spain": ["spain", "spanish"],
    "france": ["france", "french"],
    "germany": ["germany", "german"],
    "argentina": ["argentina", "argentinian"],
    "chile": ["chile", "chilean"],
    "australia": ["australia", "australian"],
    "austria": ["austria", "austrian"],
    "south africa": ["south africa", "south african"],
    "new zealand": ["new zealand", "kiwi"],
    "israel": ["israel", "israeli"],
    "hungary": ["hungary", "hungarian"],
    "greece": ["greece", "greek"],
    "romania": ["romania", "romanian"],
    "mexico": ["mexico", "mexican"],
    "canada": ["canada", "canadian"],
    "turkey": ["turkey", "turkish", "turkiye"],
    "czech republic": ["czech republic", "czech", "czechia"],
    "slovenia": ["slovenia", "slovenian"],
    "luxembourg": ["luxembourg", "luxembourger", "luxembourgian"],
    "croatia": ["croatia", "croatian"],
    "georgia": ["georgia", "georgian"],
    "uruguay": ["uruguay", "uruguayan"],
    "england": ["england", "english", "uk", "united kingdom", "britain", "british"],
    "lebanon": ["lebanon", "lebanese"],
    "serbia": ["serbia", "serbian"],
    "brazil": ["brazil", "brazilian"],
    "moldova": ["moldova", "moldovan"],
    "morocco": ["morocco", "moroccan"],
    "peru": ["peru", "peruvian"],
    "india": ["india", "indian"],
    "bulgaria": ["bulgaria", "bulgarian"],
    "cyprus": ["cyprus", "cypriot"],
    "armenia": ["armenia", "armenian"],
    "switzerland": ["switzerland", "swiss"],
    "bosnia and herzegovina": ["bosnia and herzegovina", "bosnia", "bosnian"],
    "ukraine": ["ukraine", "ukrainian"],
    "slovakia": ["slovakia", "slovak"],
    "macedonia": ["macedonia", "north macedonia", "macedonian"],
    "china": ["china", "chinese"],
    "egypt": ["egypt", "egyptian"],
}
# end of content created with ChatGPT

# Price keywords
price_keywords = {
    "budget": ["budget", "inexpensive", "cheap"],
    "mid_range": ["mid_range", "mid_priced", "affordable"],
    "premium": ["premium", "luxury", "fine", "expensive"],
}

# Flavor keywords
flavor_keywords = {
    "fruit": [
        "berry",
        "cherry",
        "apple",
        "citrus",
        "tropical",
        "fruit",
        "blackberry",
        "raspberry",
        "fruity",
    ],
    "dry": ["dry", "crisp", "tannic", "tannins", "tannin"],
    "sweet": ["sweet", "honey", "ripe", "jam"],
    "oak": ["oak", "vanilla", "toast", "cedar", "oaky", "toasty"],
    "spice": ["spice", "pepper", "cinnamon", "clove"],
    "herbal": ["herbal", "grass", "mineral", "earth", "earthy", "herbs", "grassy"],
}
