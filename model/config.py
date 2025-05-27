"""
Файл конфигурации для модели детекции аннотаций
"""

import os
import torch

# Пути к данным и модели
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "prepare-dataset-for-training", "dataset")
OUTPUT_DIR = os.path.join(ROOT_DIR, "model", "models")

# Параметры модели
MODEL_NAME = "DeepPavlov/rubert-base-cased"  # Предобученная модель для русского языка
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры обучения
EPOCHS = 100
BATCH_SIZE = 8  # Небольшой размер для экономии памяти при работе с длинными текстами
LEARNING_RATE = 2e-5
MAX_LENGTH = 512  # Максимальная длина токенизированной последовательности
SLIDING_WINDOW = True  # Использовать скользящее окно для длинных текстов
STRIDE = 64  # Шаг скользящего окна

# Параметры предсказания
PREDICT_FILE_PATH = r"C:\PyProjects\annotation-detection\extract-text-from-pdf\extracted-text\addition\1-antipova\texts\processed_text.txt"  # Путь к файлу для предсказания (None - использовать тестовый файл)
PREDICT_OUTPUT_PATH = None  # Путь для сохранения результата (None - вывести в консоль)
ANNOTATION_LABELS = {
    "O": 0,  # Не является частью аннотации
    "B-ANN": 1,  # Начало аннотации
    "I-ANN": 2,  # Внутри аннотации
}
ID2LABEL = {v: k for k, v in ANNOTATION_LABELS.items()}
LABEL2ID = ANNOTATION_LABELS

# Добавьте параметр для ограничения длины текста при предсказании
MAX_TEXT_LENGTH_FOR_PREDICTION = 4000  # Обрабатывать только первые несколько символов текста
