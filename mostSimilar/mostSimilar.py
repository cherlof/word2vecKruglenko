import sys

import gensim

models = ['C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_0_window_2_vect_size_50.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_0_window_2_vect_size_100.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_0_window_5_vect_size_50.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_0_window_5_vect_size_100.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_0_window_10_vect_size_50.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_0_window_10_vect_size_100.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_1_window_2_vect_size_50.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_1_window_2_vect_size_100.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_1_window_5_vect_size_50.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_1_window_5_vect_size_100.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_1_window_10_vect_size_50.model',
          'C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_1_window_10_vect_size_100.model',
          ]
# Функция для поиска наиболее похожих слов и интерпретации результатов
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



def main():
    # Открытие файла для записи вывода
    with open('output.txt', 'w', encoding='utf-8') as f:
        # Перенаправление стандартного вывода в файл
        sys.stdout = f
        for model in models:
            print("----------------------------------------------------")
            print("Модель "+model+"\n")
            word2vec_model = gensim.models.Word2Vec.load(model)

            # Поиск аналогий по словам
            for word1, word2, word3 in analogies:
                result = word_analogy(word2vec_model, word1, word2, word3)
                print(f"'{word1}' - '{word2}' + '{word3}' ≈ {result}")

            find_similar_words(word2vec_model, words_to_analyze)

        # Возвращение стандартного вывода в консоль
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()