import streamlit as st
import pandas as pd
import re
import string
import nltk
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove single characters
    text = re.sub(r'\b\w\b', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Function to perform topic modeling
def perform_topic_modeling(text_data, n_topics=5, n_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(text_data)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    topic_words = {}
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        topic_words[f'Topic {topic_idx + 1}'] = [(words[i], topic[i]) for i in topic.argsort()[-n_words:]]
    
    return topic_words

# Streamlit app design
st.title('Topic Modeling App')

st.write('Upload a CSV file to perform topic modeling on a specific text column.')

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Select column for text processing
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select the column for topic modeling:", columns)

    if selected_column:
        # Show option for data usage
        data_usage_option = st.selectbox("Choose data percentage for topic modeling:", 
                                         ["Full Data", "50% of Data", "20% of Data", "10% of Data", "1% of Data"])

        # Apply selected data usage
        if data_usage_option == "50% of Data":
            df = df.sample(frac=0.5, random_state=42)
        elif data_usage_option == "20% of Data":
            df = df.sample(frac=0.2, random_state=42)
        elif data_usage_option == "10% of Data":
            df = df.sample(frac=0.1, random_state=42)
        elif data_usage_option == "1% of Data":
            df = df.sample(frac=0.01, random_state=42)

        # Display sample data
        st.write("Sample Data from the selected column:")
        st.write(df[selected_column].head(10))

        # Preprocess the text only after selection
        st.write("Text preprocessing in progress...")
        df[selected_column] = df[selected_column].astype(str).apply(preprocess_text)

        st.write("Sample Data after preprocessing:")
        st.write(df[selected_column].head(10))

        # Perform Topic Modeling
        if st.button("Perform Topic Modeling"):
            # Estimate processing time based on the amount of data
            num_documents = len(df[selected_column].dropna())
            estimated_time = num_documents / 100  # Adjust this factor based on your expected processing time

            # Countdown timer
            for i in range(int(estimated_time), 0, -1):
                st.write(f"Estimated time remaining: {i} seconds")
                time.sleep(1)  # Simulate the wait time

            st.write("Running topic modeling...")
            
            topic_results = perform_topic_modeling(df[selected_column].dropna().tolist())

            st.write("Top words for each topic (with counts):")

            # Create a list for the results
            topic_data = []
            for topic, words in topic_results.items():
                for word, count in words:
                    topic_data.append({"Word": word, "Count": int(count)})

            # Create a DataFrame from the results list
            topic_df = pd.DataFrame(topic_data)

            # Display the results in a table (only show Word and Count)
            st.write(topic_df[['Word', 'Count']])
