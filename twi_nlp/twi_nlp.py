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
nltk.download("averaged_perceptron_tagger")

# ✅ GitHub Raw URL for dataset (CORRECTED)
DATASET_URL = "https://raw.githubusercontent.com/birdcoreone/NLP-python/master/twi_nlp/data/twi_words.csv"
DEFAULT_FILEPATH = os.path.join(os.getcwd(), "data", "twi_words.csv")

def __init__(self, filepath=None):
        """Initialize Twi NLP module with a dataset accessible online."""

        if filepath is None:
            filepath = DEFAULT_FILEPATH

        if not os.path.exists(filepath):
            print(f"❌ Dataset not found locally. Attempting to download from: {DATASET_URL}")

            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                # ✅ Open the URL and write to file
                with urllib.request.urlopen(DATASET_URL) as response, open(filepath, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)

                print("✅ Download successful.")
            except Exception as e:
                print(f"❌ Download failed: {e}")

        # ✅ Load dataset if available
        if os.path.exists(filepath):
            self.df = pd.read_csv(filepath)
            self.translations = dict(zip(self.df["Twi"], self.df["English"]))
        else:
            self.df = None
            self.translations = {}
            
    def translate(self, word):
        """Translate a Twi word to English."""
        return self.translations.get(word, "Translation not found")

    def get_pos(self, word):
        """Automatically tag POS by first translating to English, then tagging with NLTK."""
        english_translation = self.translate(word)

        if english_translation == "Translation not found":
            return "POS not found"
            
        tokens = word_tokenize(english_translation)
        pos_tags = pos_tag(tokens)

        return pos_tags

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

    


