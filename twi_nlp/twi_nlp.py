import os
import pandas as pd
import urllib.request
import nltk
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import chardet

# ✅ Download necessary NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

# ✅ GitHub Raw URL for dataset (CORRECTED)
DATASET_URL = "https://raw.githubusercontent.com/birdcoreone/NLP-python/main/data/twi_words.csv"
DEFAULT_FILEPATH = "data/twi_words.csv"

def detect_encoding(filepath):
    """Detect file encoding to handle UTF-8 issues."""
    with open(filepath, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]

class TwiNLP:
    def __init__(self, filepath=None):
        """Initialize the Twi NLP module with an optional dataset."""
        
        # ✅ Print the current working directory (Debugging)
        print("📂 Current Working Directory:", os.getcwd())

        # ✅ Set default dataset path (Absolute Path)
        if filepath is None:
            filepath = os.path.join(os.getcwd(), "data", "twi_words.csv")

        # ✅ Print file path being searched
        print(f"🔍 Checking if dataset exists at: {filepath}")
        print(f"🔍 Absolute Path: {os.path.abspath(filepath)}")

        # ✅ Try loading the dataset locally first
        if not os.path.exists(filepath):
            print("❌ Dataset not found locally. Attempting to download from GitHub...")
            try:
                urllib.request.urlretrieve(DATASET_URL, filepath)
                print("✅ Download successful!")
            except urllib.error.HTTPError as e:
                print(f"⚠️ Failed to download dataset from GitHub (HTTP {e.code}). Skipping download.")
            except urllib.error.URLError:
                print("⚠️ Network error while trying to fetch dataset. Skipping download.")

        # ✅ If file is STILL missing, alert the user but DO NOT crash
        if not os.path.exists(filepath):
            print(f"❌ Dataset missing: {filepath}. Please manually place it in the `data/` folder.")
            # ✅ Instead of crashing, initialize empty variables to prevent errors
            self.df = None
            self.words = []
            self.translations = {}
            self.pos_tags = {}
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            return  # ✅ Exit gracefully

        # ✅ If dataset is found, load it
        encoding = detect_encoding(filepath)
        self.df = pd.read_csv(filepath, encoding=encoding)
        self.words = self.df['Twi'].tolist()
        self.translations = dict(zip(self.df['Twi'], self.df['English']))
        self.pos_tags = dict(zip(self.df['Twi'], self.df['POS']))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def translate(self, word):
        """Translate a Twi word to English."""
        return self.translations.get(word, "Translation not found")

    def get_pos(self, word):
        """Get the POS tag of a Twi word."""
        return self.pos_tags.get(word, "POS not found")

    def search(self, keyword):
        """Search for words containing the keyword."""
        return [word for word in self.words if keyword in word]

    def tokenize(self, text):
        """Basic tokenizer using NLTK."""
        return word_tokenize(text)

    def stem_and_lemmatize(self, twi_word):
        """Stem & Lemmatize a Twi word using English as an intermediary."""
        english_translation = self.translate(twi_word)

        if english_translation == "Translation not found":
            return {"Stemmed": "N/A", "Lemmatized": "N/A"}  # No translation found

        # ✅ Apply Stemming and Lemmatization
        stemmed = self.stemmer.stem(english_translation.lower())
        lemmatized = self.lemmatizer.lemmatize(english_translation.lower())

        return {"Stemmed": stemmed, "Lemmatized": lemmatized, "Twi Equivalent": english_translation}

    def load_dataset(self, filepath):
        """Allows users to load a dataset manually."""
        encoding = detect_encoding(filepath)
        self.df = pd.read_csv(filepath, encoding=encoding)
        self.words = self.df['Twi'].tolist()
        self.translations = dict(zip(self.df['Twi'], self.df['English']))
        self.pos_tags = dict(zip(self.df['Twi'], self.df['POS']))

# ✅ Example Usage
def main():
    twi_nlp = TwiNLP()
    print(twi_nlp.stem_and_lemmatize("yɛbɛkɔ"))

if __name__ == "__main__":
    main()
