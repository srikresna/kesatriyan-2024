from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import spacy
from textblob import TextBlob
from gensim import corpora, models
import numpy as np
from gensim.models import Word2Vec
from textstat import flesch_kincaid_grade

nlp = spacy.load('en_core_web_sm')

def get_top_words(df, n=3, stop_words=None):
    # Tokenisasi dan hitung frekuensi kata
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(df['clean_text'])
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    total_freq = word_freq.sum().sort_values(ascending=False)
    
    # Simpan n kata yang paling sering muncul dan frekuensinya
    words_freq = total_freq.nlargest(n).to_dict()

    # buat fitur dari kata-kata yang paling sering muncul
    features = set(words_freq.keys())
    features = list(features)

    return words_freq, features


# persentase kata yang paling sering muncul
def get_top_words_percentage(df, n=3, stop_words=None):
    # Tokenisasi dan hitung frekuensi kata
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(df['clean_text'])
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    total_freq = word_freq.sum().sort_values(ascending=False)
    
    # Simpan n kata yang paling sering muncul dan frekuensinya
    words_freq = total_freq.nlargest(n).to_dict()

    # buat fitur dari kata-kata yang paling sering muncul
    features = set(words_freq.keys())
    features = list(features)

    # hitung persentase kata yang paling sering muncul
    total_words = total_freq.sum()
    words_freq_percentage = {word: freq/total_words for word, freq in words_freq.items()}

    return words_freq_percentage, features

def tokenize(text, stopwords):
    # Cari semua hashtag dan hapus stopwords
    return [word for word in re.findall(r"#(\w+)", text) if word not in stopwords]

def get_top_hashtags(df, n=3, stopwords=None):
    # Pisahkan data berdasarkan label
    grouped = df.groupby('label')

    hashtags_freq = {}

    for name, group in grouped:
        # Tokenisasi dan hitung frekuensi hashtag
        vectorizer = CountVectorizer(tokenizer=lambda text: tokenize(text, stopwords))
        X = vectorizer.fit_transform(group['clean_text'])
        hashtag_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        total_freq = hashtag_freq.sum().sort_values(ascending=False)
        
        # Simpan n hashtag yang paling sering muncul dan frekuensinya
        hashtags_freq[name] = total_freq.nlargest(n).to_dict()

    # buat fitur dari hashtag yang paling sering muncul
    features = set()
    for label, freq in hashtags_freq.items():
        for hashtag in freq.keys():
            features.add(hashtag)

    features = list(features)

    return hashtags_freq, features

# NER (Named Entity Recognition) menggunakan Spacy



def get_named_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def get_top_named_entities(df, n=3):
    # Pisahkan data berdasarkan label
    grouped = df.groupby('label')

    entities_freq = {}

    for name, group in grouped:
        # Tokenisasi dan hitung frekuensi named entities
        entities = group['clean_text'].apply(get_named_entities)
        entities = pd.Series([ent for sublist in entities for ent in sublist])
        total_freq = entities.value_counts()
        
        # Simpan n named entities yang paling sering muncul dan frekuensinya
        entities_freq[name] = total_freq.nlargest(n).to_dict()

    # buat fitur dari named entities yang paling sering muncul
    features = set()
    for label, freq in entities_freq.items():
        for entity in freq.keys():
            features.add(entity)

    features = list(features)

    return entities_freq, features


# sentimen analisis data bahasa indonesia menggunakan library Sastrawi dan TextBlob bahasa indonesia 


def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def get_top_sentiments(df):
    # Pisahkan data berdasarkan label
    grouped = df.groupby('label')

    sentiments = {}

    for name, group in grouped:
        # Hitung sentimen
        sentiment = group['clean_text'].apply(get_sentiment)
        sentiments[name] = sentiment.mean()

    return sentiments

# todo
# topic modeling menggunakan LDA (Latent Dirichlet Allocation) untuk menemukan topik-topik yang ada dalam masing-masing label


def get_lda_topics(df, num_topics=10):
    # Tokenisasi teks
    texts = df['clean_text'].apply(lambda x: x.split())

    # Buat dictionary dan corpus yang diperlukan untuk LDA
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Buat model LDA
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Dapatkan topik-topik dari model LDA
    topics = lda_model.print_topics(num_words=5)

    # Dapatkan distribusi topik untuk setiap dokumen
    topic_distribution = [lda_model.get_document_topics(bow) for bow in corpus]

    # Ubah distribusi topik menjadi list of lists
    topic_lists = [[dict(td).get(i, 0) for i in range(num_topics)] for td in topic_distribution]

    # Ubah list of lists menjadi array 2D
    topic_features = np.array(topic_lists)

    return topic_features, topics
 
# word embedding menggunakan Word2Vec dari library Gensim untuk dijadikan kolom fitur dalam dataframe
from gensim.models import Word2Vec

def get_word_embeddings(df, vector_size=100, window=5, min_count=1, workers=4):
    # Tokenisasi teks
    texts = df['clean_text'].apply(lambda x: x.split())

    # Buat model Word2Vec
    model = Word2Vec(texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    # Dapatkan vektor kata untuk setiap kata yang ada di model
    word_vectors = model.wv

    # Dapatkan vektor rata-rata untuk setiap dokumen
    doc_vectors = texts.apply(lambda x: np.mean([word_vectors[word] for word in x if word in word_vectors], axis=0))

    # Ubah vektor dokumen menjadi array 2D
    doc_features = np.array(doc_vectors.tolist())

    return doc_features

# character-level features

def get_character_features(df):
    # Jumlah karakter
    df['char_count'] = df['clean_text'].apply(len)

    # Jumlah kata
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

    # Rata-rata panjang kata
    df['avg_word_length'] = df['char_count'] / df['word_count']

    # Std dev panjang kata
    df['std_word_length'] = df['clean_text'].apply(lambda x: np.std([len(word) for word in x.split()]))

    return df

# sentiment score menggunakan VADER dari library NLTK


def get_sentiment_scores(df):
    # Pisahkan data berdasarkan label
    grouped = df.groupby('label')

    sentiments = {}

    for name, group in grouped:
        # Hitung sentimen
        sentiment = group['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        sentiments[name] = sentiment.mean()

    return sentiments

# readability score menggunakan Flesch-Kincaid Grade Level


def get_readability_score(text):
    return flesch_kincaid_grade(text)

def get_readability_scores(df):
    # Pisahkan data berdasarkan label
    grouped = df.groupby('label')

    readabilities = {}

    for name, group in grouped:
        # Hitung readability score
        readability = group['clean_text'].apply(get_readability_score)
        readabilities[name] = readability.mean()

    return readabilities

# syntatic features menggunakan library SpaCy

def get_syntactic_features(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    dep_tags = [token.dep_ for token in doc]
    return pos_tags, dep_tags

def get_syntactic_features_df(df):
    # Pisahkan data berdasarkan label
    grouped = df.groupby('label')

    pos_tags = {}
    dep_tags = {}

    for name, group in grouped:
        # Hitung POS tags dan dependency tags
        pos_tags[name], dep_tags[name] = zip(*group['clean_text'].apply(get_syntactic_features))

    return pos_tags, dep_tags

# semantic features menggunakan library SpaCy

def get_semantic_features(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def get_semantic_features_df(df):
    # Pisahkan data berdasarkan label
    grouped = df.groupby('label')

    entities = {}

    for name, group in grouped:
        # Hitung named entities
        entities[name] = group['clean_text'].apply(get_semantic_features)

    return entities

# POS tagging menggunakan library SpaCy

def get_pos_tags(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return pos_tags

def get_pos_tags_df(df):
    # Pisahkan data berdasarkan label
    grouped = df.groupby('label')

    pos_tags = {}

    for name, group in grouped:
        # Hitung POS tags
        pos_tags[name] = group['clean_text'].apply(get_pos_tags)

    return pos_tags

# dependency parsing menggunakan library SpaCy

def get_dep_tags(text):
    doc = nlp(text)
    dep_tags = [token.dep_ for token in doc]
    return dep_tags

def get_dep_tags_df(df):
    # Pisahkan data berdasarkan label
    grouped = df.groupby('label')

    dep_tags = {}

    for name, group in grouped:
        # Hitung dependency tags
        dep_tags[name] = group['clean_text'].apply(get_dep_tags)

    return dep_tags




