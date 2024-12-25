import sys

import gensim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = gensim.models.Word2Vec.load("C:/Users/Honta/PycharmProjects/word2vec/fitModel/AAAmodel_sg_1_window_5_vect_size_100.model")

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
visualize_embeddings(model, words_to_visualize, method='pca', filename='SkipGram.png')

def find_similar_words(model, words):
    for word in words:
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=5)
            print(f"Наиболее похожие слова для '{word}':")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        else:
            print(f"Слово '{word}' не найдено в модели.")
        print()

# Функция для выполнения задачи на аналогии слов
# positive +, negative -
def word_analogy(model, word1, word2, word3):
    try:
        result = model.wv.most_similar(positive=[word1, word3], negative=[word2])
        return result
    except KeyError as e:
        return str(e)

# Примеры задач на аналогии слов
analogies = [
    ("король", "мужчина", "женщина"),
    ("Лондон", "Англия", "Франция"),
    ("утро", "день", "ночь"),
    ("книга", "читать", "писать")
]

words_to_analyze = ['король', 'мужчина', 'женщина', 'королева']




# Открытие файла для записи вывода
with open('SkipGram.txt', 'w', encoding='utf-8') as f:
        # Перенаправление стандартного вывода в файл
    sys.stdout = f
    print("----------------------------------------------------")
    print("Модель model_sg_1_window_5_vect_size_100.model\n")


    # Поиск аналогий по словам
    for word1, word2, word3 in analogies:
        result = word_analogy(model, word1, word2, word3)
        print(f"'{word1}' - '{word2}' + '{word3}' ≈ {result}")

    find_similar_words(model, words_to_analyze)

        # Возвращение стандартного вывода в консоль
sys.stdout = sys.__stdout__


print("Графики сохранены в файлы SkipGram.png и skipgram_tsne.png.")
