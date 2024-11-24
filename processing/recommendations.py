import json

def generate_recommendations(product):
    with open('assets/database.json') as db:
        data = json.load(db)
        if product in data:
            return data[product]['recommendations']
    return []
