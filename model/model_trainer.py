"""
Модуль для обучения модели определения аннотаций в тексте
"""

import os
import logging
import numpy as np
from typing import Dict
import json

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from model.config import (
    MODEL_NAME,
    OUTPUT_DIR,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    ID2LABEL,
    LABEL2ID
)
from model.dataset_loader import get_data_loaders

logger = logging.getLogger(__name__)

def compute_metrics(p: EvalPrediction) -> Dict:
    """
    Вычисляет метрики для оценки качества модели
    
    Args:
        p: Предсказания и метки от Trainer
        
    Returns:
        Словарь с метриками
    """
    predictions = np.argmax(p.predictions, axis=2)
    labels = p.label_ids
    
    # Отладка: форма входных данных
    logger.info(f"Метрики: форма предсказаний={predictions.shape}, форма меток={labels.shape}")
    
    # Удаляем игнорируемые индексы из предсказаний и меток
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Отладка: длины очищенных данных
    total_pred_tokens = sum(len(seq) for seq in true_predictions)
    total_label_tokens = sum(len(seq) for seq in true_labels)
    logger.info(f"Метрики: всего токенов для оценки = {total_pred_tokens} (после удаления -100)")
    
    # Вычисляем метрики, игнорируя "O" класс
    true_predictions_flat = [item for sublist in true_predictions for item in sublist]
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    
    # Статистика по классам
    label_counts = {}
    for label in true_labels_flat:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    
    logger.info(f"Распределение классов в метках: {label_counts}")
    
    # Детальные метрики по классам
    report = classification_report(
        true_labels_flat,
        true_predictions_flat,
        output_dict=True,
        zero_division=0
    )
    
    # Вывод детального отчета для отладки
    logger.info("Детальный отчет по классам:")
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            logger.info(f"  Класс {class_name}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1-score']:.4f}, support={metrics['support']}")
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels_flat, 
        true_predictions_flat, 
        average='weighted',
        labels=[ID2LABEL[1], ID2LABEL[2]]  # только B-ANN и I-ANN
    )
    
    acc = accuracy_score(true_labels_flat, true_predictions_flat)
    
    logger.info(f"Метрики: accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

class AnnotationModelTrainer:
    """Класс для обучения модели определения аннотаций"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Инициализирует обучение модели
        
        Args:
            model_name: Имя или путь к предобученной модели
        """
        self.model_name = model_name
        
        logger.info(f"Инициализация модели {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Токенизатор загружен: {model_name}")
            
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(ID2LABEL),
                id2label=ID2LABEL,
                label2id=LABEL2ID
            )
            logger.info(f"Модель загружена: {model_name}")
            
            # Отладка: вывод информации о модели
            model_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Параметры модели: {model_params:,} всего")
            
            self.model.to(DEVICE)
            logger.info(f"Модель перемещена на устройство {DEVICE}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
        
        # Получаем загрузчики данных
        try:
            self.train_loader, self.val_loader, self.test_loader = get_data_loaders(self.tokenizer)
            logger.info(f"Загрузчики данных инициализированы:")
            logger.info(f"  Train: {len(self.train_loader.dataset)} примеров, {len(self.train_loader)} батчей")
            logger.info(f"  Val: {len(self.val_loader.dataset)} примеров, {len(self.val_loader)} батчей")
            logger.info(f"  Test: {len(self.test_loader.dataset)} примеров, {len(self.test_loader)} батчей")
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise

    def train_with_trainer(self, output_dir: str = OUTPUT_DIR) -> None:
        """
        Обучает модель с использованием Trainer API
        
        Args:
            output_dir: Директория для сохранения модели
        """
        # Настраиваем аргументы обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.001,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            report_to="none",
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=1,
            logging_strategy="epoch",
        )
        
        logger.info(f"Настройки обучения:")
        logger.info(f"  Эпохи: {EPOCHS}")
        logger.info(f"  Размер батча: {BATCH_SIZE}")
        logger.info(f"  Learning rate: {LEARNING_RATE}")
        
        # Инициализируем Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_loader.dataset,
            eval_dataset=self.val_loader.dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        
        # Запускаем обучение
        logger.info("Начало обучения с Trainer API...")
        
        # Перехватываем возможные ошибки при обучении
        try:
            trainer.train()
            logger.info("Обучение завершено успешно")
        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}")
            raise
        
        # Сохраняем модель
        try:
            trainer.save_model(output_dir)
            logger.info(f"Модель сохранена в {output_dir}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
        
        # Оцениваем модель на тестовых данных
        logger.info("Оценка модели на тестовом наборе данных...")
        try:
            test_results = trainer.evaluate(self.test_loader.dataset)
            logger.info(f"Результаты тестирования: {test_results}")
            
            # Сохраняем результаты для дальнейшего анализа
            test_results_path = os.path.join(output_dir, "test_results.json")
            with open(test_results_path, "w") as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Результаты теста сохранены в {test_results_path}")
            
            # Дополнительная проверка: получаем предсказания для тестового набора
            logger.info("Генерация детальных предсказаний для тестового набора...")
            test_predictions = trainer.predict(self.test_loader.dataset)
            
            # Сохраняем сырые предсказания
            predictions_path = os.path.join(output_dir, "test_predictions.npz")
            np.savez_compressed(
                predictions_path,
                predictions=test_predictions.predictions,
                labels=test_predictions.label_ids
            )
            logger.info(f"Сырые предсказания сохранены в {predictions_path}")
            
            # Анализ конкретных примеров с ошибками
            self._analyze_test_errors(test_predictions, output_dir)
            
        except Exception as e:
            logger.error(f"Ошибка при оценке модели на тесте: {e}")

    def _analyze_test_errors(self, test_predictions, output_dir):
        """Анализирует ошибочные предсказания на тестовых данных"""
        try:
            predictions = np.argmax(test_predictions.predictions, axis=2)
            labels = test_predictions.label_ids
            
            # Преобразуем индексы в метки
            error_examples = []
            
            for i in range(min(50, len(predictions))):  # Анализируем до 50 примеров
                pred = predictions[i]
                label = labels[i]
                
                # Получаем только значащие токены (исключая padding и special tokens)
                valid_indices = [j for j, l in enumerate(label) if l != -100]
                
                if len(valid_indices) == 0:
                    continue
                
                valid_preds = [ID2LABEL[pred[j]] for j in valid_indices]
                valid_labels = [ID2LABEL[label[j]] for j in valid_indices]
                
                # Проверяем, есть ли ошибки
                has_errors = any(p != l for p, l in zip(valid_preds, valid_labels))
                
                if has_errors:
                    # Получаем текст из датасета
                    input_ids = self.test_loader.dataset[i]["input_ids"]
                    text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    
                    # Считаем метрики для этого примера
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        valid_labels, 
                        valid_preds, 
                        average='weighted',
                        labels=["B-ANN", "I-ANN"],
                        zero_division=0
                    )
                    
                    # Собираем информацию о примере с ошибками
                    example_info = {
                        "id": i,
                        "text_preview": text[:100] + "..." if len(text) > 100 else text,
                        "metrics": {
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1": float(f1)
                        },
                        "errors": []
                    }
                    
                    # Добавляем конкретные ошибки
                    for j, (p, l) in enumerate(zip(valid_preds, valid_labels)):
                        if p != l:
                            token_position = valid_indices[j]
                            token = self.tokenizer.decode([input_ids[token_position]])
                            example_info["errors"].append({
                                "position": int(token_position),
                                "token": token,
                                "predicted": p,
                                "actual": l
                            })
                    
                    error_examples.append(example_info)
            
            # Сохраняем информацию об ошибках
            errors_path = os.path.join(output_dir, "test_errors.json")
            with open(errors_path, "w", encoding="utf-8") as f:
                json.dump(error_examples, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Анализ ошибок: найдено {len(error_examples)} примеров с ошибками")
            logger.info(f"Детали ошибок сохранены в {errors_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при анализе тестовых ошибок: {e}")

    def save_model(self, output_dir: str) -> None:
        """
        Сохраняет модель и токенизатор
        
        Args:
            output_dir: Директория для сохранения
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем модель
        self.model.save_pretrained(output_dir)
        
        # Сохраняем токенизатор
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Модель и токенизатор сохранены в {output_dir}")

    def load_model(self, model_dir: str) -> None:
        """
        Загружает модель и токенизатор
        
        Args:
            model_dir: Директория с моделью
        """
        # Загружаем модель
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(DEVICE)
        
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        logger.info(f"Модель и токенизатор загружены из {model_dir}")

def train_model() -> None:
    """Обучает модель определения аннотаций"""
    trainer = AnnotationModelTrainer()
    trainer.train_with_trainer()
        
if __name__ == "__main__":
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Запускаем обучение
    train_model() 