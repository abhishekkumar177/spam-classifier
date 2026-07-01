import nltk

def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)
