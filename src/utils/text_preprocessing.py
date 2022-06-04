import re
import string
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from spellchecker import SpellChecker


spell = SpellChecker()


def remove_urls(text: str) -> str:
    """Removes the urls starting with http or https or www.

    :param text: text for cleaning
    :return: the text without the urls
    """
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_html(text: str) -> str:
    """Removes the html parts.

    :param text: text for cleaning
    :return: the text without the html parts
    """
    html=re.compile(r"<.*?>")
    return html.sub(r"", text)


def remove_punctuation(text: str) -> str:
    """Removes the punctuation from the given text.

    :param text: text for cleaning
    :return: the text without punctuation
    """
    table=str.maketrans("", "", string.punctuation)
    return text.translate(table)


def remove_email_addresses(text: str) -> str:
    """Removes the email addresses from the given text.

    :param text: text for cleaning
    :return: the text without the emails that it might have
    """
    emails = re.compile(r"\S*@\S*\s?")
    return emails.sub(r"", text)


def remove_digits(text: str) -> str:
    """Removes the numbers from the given text.

    :param text: text for cleaning
    :return: the text without the digits that it might have
    """
    pattern = r'[0-9]'

    # Match all digits in the string and replace them with an empty string
    return re.sub(pattern, '', text)


def correct_spellings(text: str) -> str:
    """Corrects the text from misspelling errors

    :param text: text for cleaning
    :return: the text without the misspelling errors that it might have
    """

    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


def preprocess_text(text: str) -> str:
    """It applies all the text pre-processing functions to the text.

    :param text:
    :return:
    """
    text = text.lower()
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_email_addresses(text)
    text = remove_digits(text)
    text = remove_punctuation(text)
    text = correct_spellings(text)
    text = remove_stopwords(text)
    text = text.strip()

    return text
