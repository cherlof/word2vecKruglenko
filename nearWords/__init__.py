import gensim

# Загрузка модели CBOW
model = gensim.models.Word2Vec.load("C:/Users/Honta/PycharmProjects/word2vec/fitModel/model_sg_0_window_5_vect_size_100.model")

result1 = model.wv.most_similar(positive=["человек", "изобретатель"])
result2 = model.wv.most_similar(positive=["китай", "ребёнок"])

print(result1)
print("\n")
print(result2)


