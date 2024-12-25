import json
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')

# Функция для предобработки текста
def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации
    text = re.sub(r'[^\w\s]', '', text)
    # Токенизация
    tokens = word_tokenize(text)
    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)  # Объединяем токены обратно в строку

# Чтение и предобработка данных из JSONL файла
preprocessed_data = []
with open('/corpus.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        item = json.loads(line.strip())
        processed_text = preprocess_text(item['text'])
        preprocessed_data.append({
            'id': item['id'],
            'ProcessedText': processed_text
        })

# Сохранение предобработанного текста в CSV
with open('preprocessed_large_file.csv', 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['ID', 'ProcessedText'])
    for item in preprocessed_data:
        writer.writerow([item['id'], item['ProcessedText']])

print("Предобработка завершена и данные сохранены в файл 'preprocessed_large_file.csv'")
