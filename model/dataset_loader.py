"""
Модуль для загрузки и подготовки датасета аннотаций
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

from model.config import (
    DATASET_DIR, 
    MODEL_NAME, 
    MAX_LENGTH, 
    BATCH_SIZE, 
    SLIDING_WINDOW,
    STRIDE,
    ANNOTATION_LABELS
)

logger = logging.getLogger(__name__)

class AnnotationDataset(Dataset):
    """Датасет для обучения модели выделения аннотаций"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, dataset_type: str = "unknown"):
        """
        Инициализирует датасет для обучения/оценки модели.
        
        Args:
            data: Список словарей с данными (текст, аннотация, start_idx, end_idx)
            tokenizer: Токенизатор из библиотеки transformers
            max_length: Максимальная длина последовательности
            dataset_type: Тип датасета (train, val, test) для отладки
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_type = dataset_type
        
        # Отладка: вывод статистики о датасете
        self._log_dataset_stats()
    
    def _log_dataset_stats(self):
        """Выводит статистику о датасете для отладки"""
        total_samples = len(self.data)
        annotation_lengths = [item["end_idx"] - item["start_idx"] for item in self.data]
        text_lengths = [len(item["text"]) for item in self.data]
        
        if total_samples > 0:
            avg_ann_len = sum(annotation_lengths) / total_samples
            max_ann_len = max(annotation_lengths)
            min_ann_len = min(annotation_lengths)
            
            avg_text_len = sum(text_lengths) / total_samples
            max_text_len = max(text_lengths)
            min_text_len = min(text_lengths)
            
            logger.info(f"[{self.dataset_type}] Датасет статистика:")
            logger.info(f"[{self.dataset_type}] Всего примеров: {total_samples}")
            logger.info(f"[{self.dataset_type}] Длина аннотаций: среднее={avg_ann_len:.2f}, мин={min_ann_len}, макс={max_ann_len}")
            logger.info(f"[{self.dataset_type}] Длина текстов: среднее={avg_text_len:.2f}, мин={min_text_len}, макс={max_text_len}")
            
            # Проверка на примеры без аннотаций
            no_annotation = [i for i, item in enumerate(self.data) if item["start_idx"] == item["end_idx"]]
            if no_annotation:
                logger.warning(f"[{self.dataset_type}] Найдено {len(no_annotation)} примеров без аннотаций (start_idx == end_idx)")
                
            # Проверка на некорректные индексы аннотаций
            invalid_indices = [i for i, item in enumerate(self.data) 
                              if item["start_idx"] < 0 or item["end_idx"] > len(item["text"]) or item["start_idx"] >= item["end_idx"]]
            if invalid_indices:
                logger.error(f"[{self.dataset_type}] Найдено {len(invalid_indices)} примеров с некорректными индексами аннотаций")
                for i in invalid_indices[:3]:  # Показываем несколько примеров
                    item = self.data[i]
                    logger.error(f"Пример #{i}: start_idx={item['start_idx']}, end_idx={item['end_idx']}, text_len={len(item['text'])}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Получает элемент по индексу и преобразует его в тензоры"""
        item = self.data[idx]
        text = item["text"]
        
        # Дополнительная отладка для длинных текстов
        if len(text) > self.max_length * 4:  # текст значительно длиннее максимальной длины
            logger.debug(f"[{self.dataset_type}] Пример #{idx}: длинный текст ({len(text)} символов)")
        
        # Преобразуем текст в токены
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True
        )
        
        # Создаем метки для каждого токена (O, B-ANN, I-ANN)
        labels = [0] * len(encodings["input_ids"])  # По умолчанию "O" 
        
        # Получаем отображение смещений токенов
        offset_mapping = encodings.pop("offset_mapping")
        
        # Помечаем токены, которые являются частями аннотации
        start_idx = item["start_idx"]
        end_idx = item["end_idx"]
        
        # Отладка: проверка индексов аннотации
        if start_idx >= end_idx:
            logger.warning(f"[{self.dataset_type}] Пример #{idx}: start_idx ({start_idx}) >= end_idx ({end_idx})")
        
        # Проверяем, входит ли аннотация в токенизированный текст
        annotation_tokens_found = False
        first_match = True
        for i, (start, end) in enumerate(offset_mapping):
            # Проверяем, находится ли токен внутри аннотации
            if start >= start_idx and end <= end_idx and end > 0:  # end > 0 чтобы исключить специальные токены
                annotation_tokens_found = True
                if first_match:
                    labels[i] = ANNOTATION_LABELS["B-ANN"]  # Начало аннотации
                    first_match = False
                else:
                    labels[i] = ANNOTATION_LABELS["I-ANN"]  # Внутри аннотации
        
        # Отладка: если аннотация не найдена в токенизированном тексте
        if not annotation_tokens_found and start_idx < end_idx:
            logger.warning(f"[{self.dataset_type}] Пример #{idx}: аннотация ({start_idx}:{end_idx}) не найдена в токенизированном тексте")
            # Выведем первые несколько токенов для отладки
            for i in range(min(10, len(offset_mapping))):
                logger.debug(f"Токен #{i}: {offset_mapping[i]}")
        
        # Отладка: посчитаем количество токенов аннотации
        ann_tokens_count = sum(1 for label in labels if label > 0)
        if ann_tokens_count > 0:
            logger.debug(f"[{self.dataset_type}] Пример #{idx}: найдено {ann_tokens_count} токенов аннотации")
        
        # Преобразуем в тензоры
        return {
            "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encodings["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def load_json_data(file_path: str) -> List[Dict]:
    """Загружает данные из JSON файла"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Загружено {len(data)} примеров из {file_path}")
            
            # Проверка структуры данных
            if data and isinstance(data, list):
                sample = data[0]
                logger.info(f"Структура данных: {', '.join(sample.keys())}")
                
                # Проверка наличия необходимых ключей
                required_keys = ["text", "annotation", "start_idx", "end_idx"]
                missing_keys = [key for key in required_keys if key not in sample]
                if missing_keys:
                    logger.error(f"Отсутствуют необходимые ключи: {missing_keys}")
            return data
    except FileNotFoundError:
        logger.error(f"Файл не найден: {file_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Ошибка декодирования JSON: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из {file_path}: {e}")
        return []

def load_datasets(tokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Загружает и подготавливает наборы данных для обучения, валидации и тестирования
    
    Args:
        tokenizer: Токенизатор из библиотеки transformers
        
    Returns:
        Кортеж из трех датасетов: train, validation, test
    """
    # Пути к файлам
    train_path = os.path.join(DATASET_DIR, "train.json")
    val_path = os.path.join(DATASET_DIR, "val.json")
    test_path = os.path.join(DATASET_DIR, "test.json")
    
    logger.info(f"Загрузка данных из: {DATASET_DIR}")
    
    # Проверка существования файлов
    if not os.path.exists(train_path):
        logger.error(f"Файл не найден: {train_path}")
    if not os.path.exists(val_path):
        logger.error(f"Файл не найден: {val_path}")
    if not os.path.exists(test_path):
        logger.error(f"Файл не найден: {test_path}")
    
    # Загрузка данных
    train_data = load_json_data(train_path)
    val_data = load_json_data(val_path)
    test_data = load_json_data(test_path)
    
    logger.info(f"Загружено примеров: train - {len(train_data)}, val - {len(val_data)}, test - {len(test_data)}")
    
    # Создание датасетов
    train_dataset = AnnotationDataset(train_data, tokenizer, MAX_LENGTH, dataset_type="train")
    val_dataset = AnnotationDataset(val_data, tokenizer, MAX_LENGTH, dataset_type="val")
    test_dataset = AnnotationDataset(test_data, tokenizer, MAX_LENGTH, dataset_type="test")
    
    return train_dataset, val_dataset, test_dataset

def get_data_loaders(tokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Получает загрузчики данных для обучения, валидации и тестирования
    
    Args:
        tokenizer: Токенизатор из библиотеки transformers
    
    Returns:
        Кортеж из трех загрузчиков: train_loader, val_loader, test_loader
    """
    train_dataset, val_dataset, test_dataset = load_datasets(tokenizer)
    
    logger.info(f"Создание DataLoader с batch_size={BATCH_SIZE}")
    
    # Создание загрузчиков данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Отладочная информация: проверка структуры данных в загрузчиках
    logger.info(f"Проверка структуры DataLoader")
    
    # Проверка train_loader
    try:
        batch = next(iter(train_loader))
        logger.info(f"Train loader batch keys: {list(batch.keys())}")
        logger.info(f"Train loader batch shapes: input_ids={batch['input_ids'].shape}, "
                   f"attention_mask={batch['attention_mask'].shape}, labels={batch['labels'].shape}")
        
        # Проверка содержимого labels
        unique_labels = torch.unique(batch['labels'])
        logger.info(f"Train loader unique labels: {unique_labels.tolist()}")
    except Exception as e:
        logger.error(f"Ошибка при проверке train_loader: {e}")
    
    return train_loader, val_loader, test_loader

# Функция для диагностики файла примера для предсказания
def diagnose_predict_file(file_path: str) -> None:
    """
    Диагностирует файл для предсказаний
    
    Args:
        file_path: Путь к файлу для предсказаний
    """
    if file_path is None:
        logger.warning("Путь к файлу для предсказаний не указан (PREDICT_FILE_PATH=None)")
        return
        
    if not os.path.exists(file_path):
        logger.error(f"Файл для предсказаний не существует: {file_path}")
        return
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"Файл для предсказаний: {file_path}")
        logger.info(f"Размер текста: {len(text)} символов")
        
        # Проверяем первые 500 символов
        preview = text[:500].replace('\n', ' ').strip()
        logger.info(f"Начало текста: {preview}...")
        
        # Поиск потенциальной аннотации в начале текста
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if non_empty_lines:
            first_paragraph = non_empty_lines[0]
            logger.info(f"Первый абзац ({len(first_paragraph)} символов): {first_paragraph[:100]}...")
            
            if len(non_empty_lines) > 1:
                second_paragraph = non_empty_lines[1]
                logger.info(f"Второй абзац ({len(second_paragraph)} символов): {second_paragraph[:100]}...")
        
    except Exception as e:
        logger.error(f"Ошибка при чтении файла для предсказаний: {e}")