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

def corpus_func(df, column):
    '''
    create a textcorpus from pd.series

    Parameters
    ----------
    text, string

    Return
    ------
    concatinated string with marker ##### as selector
    '''

    return "######".join(text for text in df[column])


def preprocess_dataframe(df):
    '''
    create new columns in the data frame
    new_column: 'text' => cleaned stopwords (english)
                'text_clean' => regex, lowercase
                'text_lemma' => lemmetized
    param: df
    returns: df with new columns
    '''

    corpus = corpus_func(df, 'question')
    text_corpus = stopword_text(corpus)
    df['text_stopwords'] = text_corpus.split('######')

    clean_corpus = clean_text(text_corpus)
    df['text_clean'] = clean_corpus.split('######')

    lemma = lem_text(clean_corpus)
    df['text_lemma'] = lemma.split('######')
    return df


def clean_text(text):
    """
    Regex cleaning of the text. Filters everything except alphanumerical and '.
    Return is turned into lower case

    Parameters
    ----------
    text : string
        text to be cleaned

    Returns
    -------
    string
        lower case regex cleaned text

    """
    text = text.replace("´", "'")

    digi_punct = "[^a-zA-Z.1234567890'# ]"
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


def main():
    """
    Testing some function in file
    """


if __name__ == '__main__':
    main()