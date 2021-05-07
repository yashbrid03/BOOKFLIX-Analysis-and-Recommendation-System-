from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pickle
import pandas as pd
df = pd.read_csv("reco.csv")

book_pivot1 = df.pivot_table(
    columns='user_id', index='book_title', values='rating')
book_pivot1.fillna(0, inplace=True)

book_sparse1 = csr_matrix(book_pivot1)

model = NearestNeighbors(algorithm='brute')  # model

model.fit(book_sparse1)

filename = 'collaborative_model.pkl'
pickle.dump(model, open(filename, 'wb'))
