 
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Function to preprocess input word
def preprocess_word(word):
    return ''.join(sorted(set(word.lower())))

# Load the pre-trained model and vocabulary
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(random_state=42))
model.set_params(**{'randomforestclassifier__max_depth': None,
                    'randomforestclassifier__n_estimators': 100,
                    'tfidfvectorizer__ngram_range': (1, 2)})

# Sample bilingual dictionary file path
bilingual_dict_path = 'english_arabic_dictionary.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(bilingual_dict_path, names=['English', 'Arabic'])

# Shuffle the data
df = df.sample(frac=1, random_state=42)

# Split the data into training and testing sets
english_train, english_test, arabic_train, arabic_test = train_test_split(
    df['English'].tolist(), df['Arabic'].tolist(), test_size=0.1, random_state=42
)

# Train the model
model.fit(english_train, arabic_train)

# Streamlit app
st.title("English to Arabic Translator")

# Input box for English word
english_word = st.text_input("Enter an English word:")
 

# Translate button
if st.button("Translate"):
    # Predict translation
    translation = model.predict([english_word])[0]
    if translation == "تاريخ":
        st.success("The word entered is not available in my dictionary ! please try another word.")
    else:
        st.success(f"Translation of '{english_word}': {translation}")
