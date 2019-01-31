# --------------
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# Code starts here
dataset = fetch_20newsgroups(shuffle=True, random_state=1 , remove=('headers', 'footers', 'quotes'))
documents = dataset.data
news_df = pd.DataFrame({'document': documents})
print(news_df.head())
# Code ends here


# --------------
# Code starts here
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")

news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# make all text lowercase
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

print(news_df.clean_doc[4][5:11])

# Code ends here


# --------------
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# Code starts here
tokenized_doc= news_df['clean_doc'].apply(lambda x:x.lower().split())
tokenized_doc=tokenized_doc.apply(lambda x:[i for i in x if i not in stop_words])
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc
print(news_df['clean_doc'])
# Code ends here


# --------------
from sklearn.feature_extraction.text import TfidfVectorizer


# Code starts here
vectorizer = TfidfVectorizer(stop_words='english',max_features= 1000,max_df = 0.5 , smooth_idf=True)
X =vectorizer.fit_transform(news_df['clean_doc'])
# Code ends here


# --------------
from sklearn.decomposition import TruncatedSVD

# Code starts here
doc_complete = news_df['clean_doc'].tolist()

svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100 , random_state = 122)

svd_model.fit(X)
print(svd_model.components_)

terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")

# Code ends here


# --------------
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim.models.ldamodel import LdaModel
import pprint
from gensim import corpora


# Code starts here
doc_complete = news_df['clean_doc'].tolist()
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    doc = doc.lower().split()
    stop_free = " ".join([i for i in doc if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

dictionary = corpora.Dictionary(doc_clean)

# Creating the corpus
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# Creating the LDA model
ldamodel = LdaModel(corpus=doc_term_matrix, num_topics=5,id2word=dictionary, random_state=20, passes=30)

# printing the topics
# pprint(ldamodel.print_topics())
print(ldamodel.print_topics(num_topics=5, num_words=3))
# Code ends here


