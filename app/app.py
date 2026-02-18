import streamlit as st
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "category_model.pkl"), "rb"))
priority_model = pickle.load(open(os.path.join(BASE_DIR, "priority_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

templates = pd.read_csv(os.path.join(BASE_DIR, "reply_templates.csv"))

def generate_reply(category, priority):
    matches = templates[
        (templates["category"] == category) &
        (templates["priority"] == priority)
    ]

    if len(matches) == 0:
        return "Thank you for contacting support."

    return matches.sample(1)["reply"].values[0]

st.title("AI Customer Support Assistant")

user_input = st.text_area("Enter customer message:")

if st.button("Generate Response"):
    vec = vectorizer.transform([user_input])

    category = model.predict(vec)[0]
    priority = priority_model.predict(vec)[0]

    reply = generate_reply(category, priority)

    st.write("### Predicted Category:", category)
    st.write("### Predicted Priority:", priority)
    st.write("### Suggested Reply:")
    st.success(reply)