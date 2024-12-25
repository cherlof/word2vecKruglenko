import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import gensim

model = gensim.models.Word2Vec.load("C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_1_window_5_vect_size_100.model")

vocab = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in vocab])

top_n = 2000
vocab = vocab[:top_n]
word_vectors = word_vectors[:top_n]

pca = PCA(n_components=2)
pca_result = pca.fit_transform(word_vectors)

fig = px.scatter(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    text=vocab,
    title=f"PCA SKIP (Size={top_n})",
    labels={'x': 'comp1', 'y': 'comp2'}
)
fig.update_traces(textposition='top center', marker=dict(size=5))
fig.write_html("skipGram.html")