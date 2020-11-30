import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# default setting for lemmatizer and stopwords
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# remove stop words
nltk.download('stopwords')
my_stopwords = set(stopwords.words('english'))


def clean_text(text):
    """
    Regex cleaning of the text. Filters everything except alphanumerical and '.
    Return is turned into lower case

    Parameters
    ----------
    text : string
        Pixel size of the height of a "noise image"

    Returns
    -------
    string
        lower case regex cleaned text

    """
    text = text.replace("´", "'")

    digi_punct = "[^a-zA-Z.1234567890' ]"
    text = re.sub(digi_punct, " ", text)
    text = " ".join(text.split())
    text = text.lower()
    return text



def stopword_text(text):
    """
    Remove all words in the text that are in the stopword list
     
    Parameters
    ----------
    text : string
    
    Returns
    -------
    string
        text only stopwords
     
    """

    return " ".join([word for word in text.split() if word not in my_stopwords])



def lem_text(text):
    """
    Group the different inflected forms of a word so they can be analysed as 
    a single item
     
    Parameters
    ----------
    text : string
    
    Returns
    -------
    string
        text with lemmas
    """

    lem_sentence = text.split()
    for i, word in enumerate(text.split()):
        for pos in "n", "v", "a", "r":
            lem = lemmatizer.lemmatize(word, pos=pos)
            if lem != word:
                lem_sentence[i] = lem
                break
            else:
                lem_sentence[i] = word
    return " ".join(lem_sentence)
