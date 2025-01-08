import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Initialize necessary tools
# Download the 'punkt_tab' resource using nltk.download()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab') # Download the missing 'punkt_tab' resource


spell = SpellChecker()

def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) or word for word in words]  # Keep original word if correction is None
    return " ".join(corrected_words)

def preprocess_text(text):
    # Step 1: Spelling Correction
    text = correct_spelling(text)
    
    # Step 2: Lowercase conversion
    text = text.lower()
    
    # Step 3: Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Step 4: Tokenization
    words = word_tokenize(text)
    
    # Step 5: Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    # Step 6: Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)