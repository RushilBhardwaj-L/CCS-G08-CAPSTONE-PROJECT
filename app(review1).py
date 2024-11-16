import openai
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Set OpenAI API Key (replace with your actual OpenAI API key)
openai.api_key = 'use OPENAI key'

# Streamlit UI setup
st.title("Financial Misinformation Detector")
st.write("Enter financial news below, and the system will check its accuracy and provide supporting data.")

# Text input for the news article
input_news = st.text_area("Enter the financial news you want to check:")

# Function to query GPT-4 for real-time misinformation detection and supporting data
def query_gpt4(news_text):
    # GPT-4 API request for real-time accuracy check and supporting data
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial expert tasked with checking the accuracy of financial news and providing supporting data or sources."},
            {"role": "user", "content": f"Is this financial news accurate? Please provide supporting details and relevant context for: {news_text}, Answers with either accurate or inaccurate"}
        ]
    )
    
    # Extract GPT-4's response
    gpt_output = response['choices'][0]['message']['content']
    st.write(gpt_output)
    
    # Analyze GPT-4's response to determine prediction (True/False) and extract evidence
    prediction = None
    if "ACCURATE" in gpt_output or "Accurate" in gpt_output or "accurate" in gpt_output:
        prediction = 1  # True news
    elif "INACCURATE" in gpt_output or "Inaccurate" in gpt_output or "inaccurate" in gpt_output:
        prediction = 0  # False news
    else:
        prediction = None  # Unable to determine
        
    return prediction, gpt_output

# Initialize a placeholder for predictions and evidence
predictions = []
supporting_evidence = []

# Check the news when the button is pressed
if st.button("Check Accuracy"):
    if input_news:
        # Query GPT-4 with the input news
        prediction, evidence = query_gpt4(input_news)
        
        # Display results based on GPT-4's output
        if prediction is not None:
            if prediction == 1:
                st.success("The news is likely accurate.")
            else:
                st.error("The news is likely false.")
            
            # Show supporting evidence from GPT-4
            st.subheader("Supporting Evidence from GPT-4:")
            st.write(evidence)
        else:
            st.warning("Unable to determine the accuracy of the news. Please check again or verify with other sources.")
    else:
        st.warning("Please enter some news text for evaluation.")
