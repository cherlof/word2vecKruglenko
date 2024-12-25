# Word2vec
## Данные: датасет qa_wiki_ru https://huggingface.co/datasets/AIR-Bench/qa_wiki_ru
### Задание
#### Подготовка данных:
• Соберите или найдите объемный текстовый корпус (например,
коллекции статей, книг или доменно-специфических текстов).
  
• Выполните предобработку текста (токенизация, приведение к нижнему
регистру, удаление пунктуации и стоп-слов при необходимости).  
##### Package textEdit.\_\_init__.py при помощи библиотеки nltk предобрабатывает датасет и сохраняет [preprocessed_large_file.csv](textEdit/preprocessed_large_file.csv).  
#### Обучение модели:
• С помощью библиотеки “gensim” обучите модели Word2Vec используя
архитектуры CBOW и Skip-gram на вашем корпусе.  

• Поэкспериментируйте с различными гиперпараметрами (размер
эмбеддинга, размер окна, минимальная частота и т. д.).  
  
Произведено обучение с гиперпараметрами sg = [0, 1], window = [2, 5, 10], vector_size = [50, 100], где sg отвечает за использование
CBOW SkipGram, window - размер окна, vector_size - размер эмбеддинга.  
Сохранено 12 моделей и протестированы на различных тестовых данных.  
Результат в [Лог](mostSimilar/output.txt)
  
##### Обучение происходит в Package fitModel.\_\_init__.py с различными гиперпараметрами такими как window, vector_size, sg(0-CBOW,1-SkipGram) и сохраняет модели в файлы. 

#### Анализ:
• Визуализируйте эмбеддинги слов с помощью методов PCA или t-SNE.
• Найдите наиболее похожие слова для заданного списка слов и
интерпретируйте результаты.
• Выполните задачи на аналогии слов (например, "король" - "мужчина" +
"женщина" ≈ ?).  
  [Лог](mostSimilar/output.txt)  
  
Визуализация эмбедингов(На лучшем наборе гиперпараметров)
### CBOW  
![cbow_pca.png](showResult/cbow_pca.png)
  
![cbow_pca.png](showResult/cbow_pca.png)
### SkipGram  
![SkipGram.png](showResult/SkipGram.png)
  
![SkipGram.png](showResult/SkipGram.png)