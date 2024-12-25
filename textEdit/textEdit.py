import json
import csv
import re
import nltk
nltk.download("omw-1.4")
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Загрузка необходимых ресурсов NLTK

stop_words = set(stopwords.words("russian"))
lemmatizer = WordNetLemmatizer()


# Функция для предобработки текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яё\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Чтение и предобработка данных из JSONL файла
preprocessed_data = []
with open('C:\\Users\\Honta\\PycharmProjects\\word2vec\\corpus.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        item = json.loads(line.strip())
        processed_text = preprocess_text(item['text'])
        preprocessed_data.append({
            'id': item['id'],
            'ProcessedText': processed_text
        })
    print("Файл открыт")



# Сохранение предобработанного текста в CSV
with open('preprocessed_large_file.csv', 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['ID', 'ProcessedText'])
    for item in preprocessed_data:
        writer.writerow([item['id'], item['ProcessedText']])

print("Предобработка завершена и данные сохранены в файл 'preprocessed_large_file.csv'")
