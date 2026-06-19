 import json
import random
import nltk

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download tokenizer
nltk.download('punkt')

# ==========================
# Load Intent Dataset
# ==========================

with open('intent_dataset.json', 'r') as file:
    data = json.load(file)

texts = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])

# ==========================
# TF-IDF Vectorization
# ==========================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

# ==========================
# Train Naive Bayes Model
# ==========================

model = MultinomialNB()

model.fit(X, labels)

# ==========================
# Rule-Based Responses
# ==========================

rule_based = {
    "help": "I can help you with courses, greetings, thanks and general queries.",
    "contact": "You can contact us at support@example.com.",
    "hours": "Our support team is available from 9 AM to 6 PM."
}

# ==========================
# Chat Function
# ==========================

def chatbot_response(user_input):

    user_input_lower = user_input.lower()

    # Rule-Based Part
    for keyword in rule_based:
        if keyword in user_input_lower:
            return rule_based[keyword]

    # ML-Based Part
    input_vector = vectorizer.transform([user_input])

    prediction = model.predict(input_vector)[0]

    for intent in data['intents']:
        if intent['tag'] == prediction:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand."

# ==========================
# Chat Loop
# ==========================

print("=" * 50)
print("Hybrid Chatbot Started")
print("Type 'quit' to exit")
print("=" * 50)

while True:

    user_text = input("You: ")

    if user_text.lower() == "quit":
        print("Bot: Goodbye!")
        break

    response = chatbot_response(user_text)

    print("Bot:", response)
