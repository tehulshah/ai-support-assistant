from openai import OpenAI
import streamlit as st
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "category_model.pkl"), "rb"))
priority_model = pickle.load(open(os.path.join(BASE_DIR, "priority_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

templates = pd.read_csv(os.path.join(BASE_DIR, "reply_templates.csv"))

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_reply(message, category, priority):

    prompt = f"""
You are a professional customer support assistant.

Ticket category: {category}
Ticket priority: {priority}

Customer message:
{message}

Generate a polite, helpful support reply.
Keep it concise and professional.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    return response.choices[0].message.content

st.title("AI Customer Support Assistant")

user_input = st.text_area("Enter customer message:")

if st.button("Generate Response"):
    vec = vectorizer.transform([user_input])

    category = model.predict(vec)[0]
    priority = priority_model.predict(vec)[0]

    reply = generate_reply(user_input, category, priority)

    st.write("### Predicted Category:", category)
    st.write("### Predicted Priority:", priority)
    st.write("### Suggested Reply:")
    st.success(reply)