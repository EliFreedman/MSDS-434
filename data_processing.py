from collections import Counter
import math
from urllib.parse import urlparse
import tldextract
import re
import pandas as pd


def shannon_entropy(s):
    """
    Calculates the Shannon entropy of a given sequence.

    Shannon entropy is a measure of the unpredictability or information
    content in a sequence. It is commonly used in information theory to
    quantify the diversity of elements in a dataset.

    Parameters:
        s (iterable): The input sequence (e.g., string, list) whose
            entropy is to be calculated.

    Returns:
        float: The Shannon entropy value of the sequence.

    Example:
        >>> shannon_entropy("hello")
        1.9219280948873623
    """
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())


def extract_url_features(url):
    """
    Extracts a set of features from a given URL for analysis
    or machine learning tasks.
    Parameters
    ----------
    url : str
        The URL string to extract features from.
    Returns
    -------
    pandas.Series
        A Series containing the following features:
            - url_length: Length of the entire URL.
            - hostname_length: Length of the hostname part.
            - path_length: Length of the path part.
            - query_length: Length of the query string.
            - num_dots: Number of '.' characters in the URL.
            - num_hyphens: Number of '-' characters in the URL.
            - num_at: Number of '@' characters in the URL.
            - num_question_marks: Number of '?' characters in the URL.
            - num_equals: Number of '=' characters in the URL.
            - num_underscores: Number of '_' characters in the URL.
            - num_ampersands: Number of '&' characters in the URL.
            - num_digits: Number of digit characters in the URL.
            - has_https: 1 if the URL starts with 'https', 0 otherwise.
            - uses_ip: 1 if the URL uses an IP address, 0 otherwise.
            - num_subdomains: Number of subdomains in the URL.
            - has_<keyword>: 1 if the keyword (e.g., 'login', 'secure', etc.)
                is present in the URL, 0 otherwise.
            - has_port: 1 if a port is specified in the hostname, 0 otherwise.
            - url_entropy: Shannon entropy of the URL string.
    """
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query

    features = {}
    features['url_length'] = len(url)
    features['hostname_length'] = len(hostname)
    features['path_length'] = len(path)
    features['query_length'] = len(query)

    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_at'] = url.count('@')
    features['num_question_marks'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_underscores'] = url.count('_')
    features['num_ampersands'] = url.count('&')
    features['num_digits'] = sum(c.isdigit() for c in url)

    features['has_https'] = int(url.lower().startswith('https'))
    features['uses_ip'] = int(
        bool(re.search(r'http[s]?://(?:\d{1,3}\.){3}\d{1,3}', url))
    )

    features['num_subdomains'] = (
        len(ext.subdomain.split('.')) if ext.subdomain else 0
    )

    suspicious_keywords = [
        'login',
        'secure',
        'account',
        'update',
        'free',
        'lucky',
        'banking',
        'confirm'
    ]
    for keyword in suspicious_keywords:
        features[f'has_{keyword}'] = int(keyword in url.lower())

    features['has_port'] = int(':' in hostname)
    features['url_entropy'] = shannon_entropy(url)

    return pd.Series(features)
