import gensim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = gensim.models.Word2Vec.load("C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_0_window_5_vect_size_100.model")

# Функция для визуализации эмбеддингов и сохранения в файл
def visualize_embeddings(model, words, method='pca', filename='embedding_plot.png'):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2)

    reduced_vectors = reducer.fit_transform(word_vectors)

    plt.figure(figsize=(10, 7))
    for i, word in enumerate(words):
        if word in model.wv:
            plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
            plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
    plt.savefig(filename)
    plt.close()

# Пример визуализации эмбеддингов слов для CBOW модели
words_to_visualize = ['король', 'мужчина', 'женщина', 'королева']
visualize_embeddings(model, words_to_visualize, method='pca', filename='cbow_pca.png')


print("Графики сохранены в файлы cbow_pca.png и skipgram_tsne.png.")
