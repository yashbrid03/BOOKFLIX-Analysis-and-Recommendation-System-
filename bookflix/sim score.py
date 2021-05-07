import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(nrows):
    data = pd.read_csv("final_books3.csv")
    return data


books = load_data(10000)
df = books
content_data = df[['original_title', 'authors', 'average_rating', 'image_url']]
content_data = content_data.astype(str)

content_data['content'] = content_data['original_title'] + ' ' + content_data['authors'] + \
    ' ' + content_data['average_rating'] + ' ' + content_data['image_url']

content_data = content_data.reset_index()
indices = pd.Series(content_data.index, index=content_data['original_title'])

# removing stopwords
tfidf = TfidfVectorizer(stop_words='english')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(content_data['authors'])

# Output the shape of tfidf_matrix
tfidf_matrix.shape

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(content_data['content'])

cosine_sim_content = cosine_similarity(count_matrix, count_matrix)


with open('sim_score.pkl', 'wb') as f:
    pickle.dump(cosine_sim_content, f)
