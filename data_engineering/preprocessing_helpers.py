import html
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

english_stopwords = set(stopwords.words('english'))
newStopWords = {"’"}
english_stopwords = english_stopwords.union(newStopWords)
lemmatizer = WordNetLemmatizer()


def remove_punctuation_and_lower(text: str) -> str:
    "Remove punctuation and lower text."

    text = text.translate(str.maketrans("", "", string.punctuation)) if type(text) == str else text
    text = text.replace( "\\", "") if type(text) == str else text

    return text.lower() if type(text) == str else text


def tokenize(text: str) -> list:
    "Tokenize if text is string"

    return word_tokenize(text) if type(text) == str else None


def remove_stopwords(tokens) -> list:
    "Return text without stopwords."

    return [t for t in tokens if t not in english_stopwords] if tokens else None


def lemmatize(tokens: list) -> list:
    "Lemmatize all tokens."
    
    do_not_lemmatize = ("us", "vs")
    return [lemmatizer.lemmatize(token) if token not in do_not_lemmatize else token for token in tokens] if tokens else None


def classic_nlp_preprocessing(df: pd.DataFrame, columns: list, remove_stop_words: bool=True, do_lemmatization: bool=True, newsSpace: bool=False) -> pd.DataFrame:
    """
    Perform the following step for the specified columns on the specified DataFrame (in order):
    - Filter out rows without a title and where the date is 0000-00-00 00:00:00 or does not match the pattern
      if newsSpace is True (special feature for the newsSpace dataset)
    - Remove punctuation
    - Transform to lower case
    - Tokenize
    - Remove stopwords
    - Lemmatize
    Do it only if the corresponding boolean is True.
    """

    for col in columns:
        assert col in df.columns, "Columns in columns list was not found in the DataFrame!"

        if newsSpace:
            df = df[df['title'].notnull()]
            df = df[df['pubdate'] != '0000-00-00 00:00:00']
            df = df[df['title'] != '0000-00-00 00:00:00']

            # Get posts where the pubdate is legitimate
            pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
            df = df[pd.notna(df['pubdate']) & df['pubdate'].str.match(pattern)]

        df[col + "_lowered"] = df[col].apply(remove_punctuation_and_lower)

        df[col + "_tokenized"] = df[col + "_lowered"].apply(tokenize)

        if remove_stop_words:
            df[col + "_removed_stopwords"] = df[col + "_tokenized"].apply(remove_stopwords)
        
        if do_lemmatization:
            df[col + "_lemmatized"] = df[col + "_removed_stopwords"].apply(lemmatize)
        else:
            df[col + "_lemmatized"] = df[col + "_tokenized"].apply(lemmatize)

    return df


def replace_html_number_codes(text: str) -> str:
    "Replace HTML number codes with Unicode characters."

    return html.unescape(text) if type(text) == str else text
