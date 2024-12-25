import pandas as pd
from gensim.models import Word2Vec
import textEdit

def main():
    sg = [0, 1]
    window = [5]
    vector_size = [100]

    # Определите размер чанка
    chunk_size = 10000  # Попробуйте различные значения для оптимизации

    # Инициализируйте пустой список для хранения данных
    corpus = []

    # Чтение файла по частям
    for chunk in pd.read_csv('preprocessed_large_file.csv', chunksize=chunk_size):
        # Обработка каждой части
        part = chunk['ProcessedText'].apply(lambda x: x.split()).tolist()
        corpus.extend(part)

    print(f"Загрузка завершена. Общий размер корпуса: {len(corpus)}")

    for s in sg:
        for w in window:
            for vs in vector_size:
                print("Обучение модели")
                # Обучение модели CBOW или Skip-gram
                model = Word2Vec(sentences=corpus, vector_size=vs, window=w, min_count=1, sg=s, workers=12)
                # Сохранение модели
                model_name = f"AAAmodel_sg_{s}_window_{w}_vect_size_{vs}.model"
                model.save(model_name)
                print(f"Модель {model_name} обучена и сохранена.")

if __name__ == "__main__":
    main()

