import streamlit as st
import requests

st.set_page_config(page_title="Spam Classifier", page_icon="📨")
st.title("📨 Spam/Ham Classifier")
st.write("Paste a message below and check if it's spam or ham.")

API_URL = "http://127.0.0.1:8000/predict"  # change this after deployment

message = st.text_area("Enter your message:", height=120)

if st.button("Classify"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        response = requests.post(API_URL, json={"text": message})
        if response.status_code == 200:
            result = response.json()
            label = result['prediction']
            confidence = result['confidence']
            if label == 'spam':
                st.error(f"🚨 SPAM ({confidence:.1%} confidence)")
            else:
                st.success(f"✅ HAM ({confidence:.1%} confidence)")
        else:
            st.error("Error calling API")
