import streamlit as st
import pickle
import pandas as pd
import os
import random

BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "category_model.pkl"), "rb"))
priority_model = pickle.load(open(os.path.join(BASE_DIR, "priority_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

templates = pd.read_csv(os.path.join(BASE_DIR, "reply_templates.csv"))


def generate_reply(message, category, priority):

    greeting = random.choice([
        "Thank you for contacting us.",
        "We appreciate you reaching out.",
        "Thanks for bringing this to our attention."
    ])

    urgency_line = {
        "High": "We understand this is urgent and are prioritizing your request.",
        "Medium": "Our support team is reviewing your request.",
        "Low": "Our team will review this shortly."
    }

    category_actions = {
        "Billing": "Our billing team is checking the transaction details.",
        "Technical": "Our technical team is investigating the issue.",
        "Account": "Our account support team is reviewing your account.",
        "General Inquiry": "We are reviewing your inquiry.",
        "Fraud": "Our security team is reviewing the case."
    }

    closing = random.choice([
        "We will keep you updated.",
        "You will receive updates shortly.",
        "We appreciate your patience."
    ])

    reply = f"""
{greeting}

Regarding your concern: "{message}"

{urgency_line.get(priority, "")}
{category_actions.get(category, "Our team is looking into this issue.")}

{closing}
"""

    return reply


st.title("AI Customer Support Assistant")

user_input = st.text_area("Enter customer message:")

if st.button("Generate Response"):
    vec = vectorizer.transform([user_input])

    category = model.predict(vec)[0]
    priority = priority_model.predict(vec)[0]

    reply = generate_reply(user_input, category, priority)

    # st.write("### Predicted Category:", category)
    # st.write("### Predicted Priority:", priority)
    st.write("### Suggested Reply:")
    st.success(reply)