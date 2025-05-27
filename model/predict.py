"""
Модуль для предсказания аннотаций в тексте
"""

import os
import logging
import torch
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any

from transformers import AutoModelForTokenClassification, AutoTokenizer

from model.config import (
    ANNOTATION_LABELS,
    DEVICE,
    OUTPUT_DIR,
    PREDICT_FILE_PATH,
    PREDICT_OUTPUT_PATH,
    SLIDING_WINDOW,
    STRIDE,
    MAX_LENGTH,
    MAX_TEXT_LENGTH_FOR_PREDICTION
)
from model.dataset_loader import diagnose_predict_file

logger = logging.getLogger(__name__)

class AnnotationPredictor:
    """Класс для предсказания аннотаций в тексте"""
    
    def __init__(self, model_dir: str = OUTPUT_DIR):
        """
        Инициализирует предсказатель аннотаций
        
        Args:
            model_dir: Директория с моделью
        """
        self.model_dir = model_dir
        
        # Проверяем существование директории с моделью
        if not os.path.exists(model_dir):
            raise ValueError(f"Директория с моделью не существует: {model_dir}")
        
        logger.info(f"Инициализация предсказателя из {model_dir}")
        
        # Загружаем модель и токенизатор
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.info(f"Токенизатор загружен успешно")
            
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            logger.info(f"Модель загружена успешно")
            
            # Проверка наличия нужных атрибутов
            if hasattr(self.model.config, 'id2label'):
                logger.info(f"Маппинг id2label: {self.model.config.id2label}")
            else:
                logger.warning(f"У модели отсутствует маппинг id2label, используется стандартный")
                
            self.model.to(DEVICE)
            self.model.eval()
            logger.info(f"Модель перемещена на устройство {DEVICE} в режиме оценки")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def _merge_overlapping_fragments(self, fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Объединяет перекрывающиеся фрагменты
        
        Args:
            fragments: Список фрагментов аннотаций
            
        Returns:
            Список объединенных фрагментов
        """
        if not fragments:
            logger.warning("Не найдено фрагментов для объединения")
            return []
            
        # Сортируем фрагменты по позиции начала
        sorted_fragments = sorted(fragments, key=lambda x: x['start'])
        merged = [sorted_fragments[0]]
        
        for current in sorted_fragments[1:]:
            previous = merged[-1]
            
            # Определяем перекрытие
            overlap = min(previous['end'], current['end']) - max(previous['start'], current['start'])
            overlap_ratio = overlap / (min(previous['end'] - previous['start'], current['end'] - current['start']) + 0.001)
            
            # Если текущий фрагмент перекрывается с предыдущим или находится близко
            if current['start'] <= previous['end'] + 50 or overlap_ratio > 0.3:
                # Сохраняем крайние позиции
                start = min(previous['start'], current['start'])
                end = max(previous['end'], current['end'])
                
                # Создаем объединенный текст
                if current['start'] < previous['start']:
                    # Текущий фрагмент начинается раньше
                    merged_text = current['text'] + previous['text'][previous['end'] - current['end']:]
                else:
                    # Предыдущий фрагмент начинается раньше или одновременно
                    merged_text = previous['text'] + current['text'][current['end'] - previous['end']:]
                
                # Вычисляем взвешенную уверенность на основе длины фрагментов
                prev_weight = (previous['end'] - previous['start']) / ((previous['end'] - previous['start']) + (current['end'] - current['start']))
                current_weight = 1 - prev_weight
                weighted_confidence = (previous['confidence'] * prev_weight) + (current['confidence'] * current_weight)
                
                # Обновляем предыдущий фрагмент
                previous['start'] = start
                previous['end'] = end
                previous['text'] = merged_text
                previous['confidence'] = weighted_confidence
            else:
                # Если нет перекрытия, добавляем новый фрагмент
                merged.append(current)
        
        # Проводим дополнительную проверку и объединяем фрагменты, которые чрезмерно близки
        i = 0
        while i < len(merged) - 1:
            current = merged[i]
            next_frag = merged[i+1]
            
            # Близкие фрагменты (менее 50 символов между ними)
            if next_frag['start'] - current['end'] < 50:
                # Объединяем фрагменты
                # Объединяем тексты с учетом промежутка
                gap_text = current['text'] + "..." + next_frag['text']
                
                # Обновляем текущий фрагмент
                current['end'] = next_frag['end']
                current['text'] = gap_text
                current['confidence'] = (current['confidence'] + next_frag['confidence']) / 2
                
                # Удаляем следующий фрагмент
                merged.pop(i+1)
            else:
                i += 1
        
        logger.info(f"Объединено {len(fragments)} фрагментов в {len(merged)}")
        return merged

    def predict(self, text: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Предсказывает аннотацию в тексте
        
        Args:
            text: Входной текст
            
        Returns:
            Кортеж из строки аннотации, списка найденных фрагментов и уверенности модели
        """
        logger.info(f"Начало предсказания для текста длиной {len(text)} символов")
        
        # Всегда используем токенизатор для разбивки текста, даже для коротких текстов
        # Это позволяет унифицировать обработку и избежать ошибок
        if SLIDING_WINDOW:
            logger.info(f"Обработка текста со скользящим окном (длина: {len(text)} символов)")
            
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                stride=STRIDE,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True
            )
            
            # Количество фрагментов
            num_chunks = len(encodings["input_ids"])
            logger.info(f"Текст разбит на {num_chunks} фрагментов")
        else:
            # Для режима без скользящего окна
            logger.info(f"Обработка текста стандартным способом (длина: {len(text)} символов)")
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True
            )
        
        # Сохраняем смещения токенов для последующего использования
        offset_mapping = encodings.pop("offset_mapping")
        
        # Получаем предсказания
        logger.info("Выполнение предсказания моделью")
        with torch.no_grad():
            try:
                # Перемещаем входные данные на устройство
                input_ids = encodings["input_ids"].to(DEVICE)
                attention_mask = encodings["attention_mask"].to(DEVICE)
                
                # Получаем выходы модели
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Получаем логиты и меры уверенности
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=2)
                
                # Получаем предсказанные классы и их вероятности
                predictions = torch.argmax(logits, dim=2).cpu().numpy()
                prediction_probs = torch.max(probabilities, dim=2)[0].cpu().numpy()
                
            except Exception as e:
                logger.error(f"Ошибка при предсказании: {e}")
                return "", [], 0.0
        
        # Обрабатываем предсказания для получения фрагментов аннотации
        fragments = []
        
        # Обрабатываем каждую последовательность
        for i in range(len(predictions)):
            fragment = self._extract_annotation_fragment(
                text,
                predictions[i],
                prediction_probs[i],
                offset_mapping[i].numpy(),
                chunk_idx=i
            )
            if fragment:
                fragments.append(fragment)
        
        logger.info(f"Извлечено фрагментов аннотаций: {len(fragments)}")
        
        # Если фрагментов нет, возвращаем пустую строку
        if not fragments:
            logger.warning("Не найдено фрагментов аннотации в тексте")
            return "", [], 0.0
        
        # Выполняем объединение перекрывающихся фрагментов
        merged_fragments = self._merge_overlapping_fragments(fragments)
        
        # Сортируем фрагменты по уверенности и позиции
        sorted_fragments = sorted(merged_fragments, 
                                 key=lambda x: (x["confidence"], -x["start"]), 
                                 reverse=True)
        
        # Предпочитаем аннотации в начале текста (первые 20% текста)
        early_fragments = [f for f in sorted_fragments 
                          if f["start"] < len(text) * 0.2]
        
        # Выбираем лучший фрагмент на основе уверенности и позиции
        if early_fragments:
            # Предпочитаем фрагменты из начала текста
            best_fragment = early_fragments[0]
            logger.info(f"Выбран фрагмент из начала текста с уверенностью {best_fragment['confidence']:.4f}")
        elif merged_fragments:
            # Если нет фрагментов в начале, берем с наивысшей уверенностью
            best_fragment = sorted_fragments[0]
            logger.info(f"Выбран фрагмент с наивысшей уверенностью {best_fragment['confidence']:.4f}")
        else:
            logger.warning("После объединения фрагментов не осталось аннотаций")
            return "", [], 0.0
            
        # Извлекаем аннотацию из текста
        annotation_text = text[best_fragment["start"]:best_fragment["end"]]
        
        logger.info(f"Лучший фрагмент: start={best_fragment['start']}, end={best_fragment['end']}, "
                   f"confidence={best_fragment['confidence']:.4f}")
        
        return annotation_text, merged_fragments, best_fragment["confidence"]

    def _extract_annotation_fragment(
        self,
        text: str,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        offset_mapping: np.ndarray,
        chunk_idx: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Извлекает фрагмент аннотации из предсказаний
        
        Args:
            text: Исходный текст
            predictions: Предсказанные классы для токенов
            probabilities: Вероятности предсказаний
            offset_mapping: Отображение между токенами и позициями в тексте
            chunk_idx: Индекс фрагмента для отладки
            
        Returns:
            Словарь с информацией о фрагменте или None, если фрагмент не найден
        """
        # Ищем начало и конец аннотации
        annotation_spans = []
        current_span = None
        
        i_ann_without_b_ann_count = 0
        
        for i, (pred, prob, (start, end)) in enumerate(zip(predictions, probabilities, offset_mapping)):
            # Пропускаем специальные токены и токены, не относящиеся к аннотации (класс "O")
            if end == 0 or pred == ANNOTATION_LABELS["O"]:
                # Если был открыт span, закрываем его
                if current_span is not None:
                    annotation_spans.append(current_span)
                    current_span = None
                continue
            
            if pred == ANNOTATION_LABELS["B-ANN"]:
                # Если был открыт span, закрываем его
                if current_span is not None:
                    annotation_spans.append(current_span)
                
                # Создаем новый span
                current_span = {
                    "start": int(start),
                    "end": int(end),
                    "probs": [float(prob)]
                }
            elif pred == ANNOTATION_LABELS["I-ANN"] and current_span is not None:
                # Расширяем текущий span
                current_span["end"] = int(end)
                current_span["probs"].append(float(prob))
            elif pred == ANNOTATION_LABELS["I-ANN"] and current_span is None:
                # Нашли I-ANN без предшествующего B-ANN - необычная ситуация
                i_ann_without_b_ann_count += 1
                # Создаем новый span, предполагая, что это продолжение аннотации
                current_span = {
                    "start": int(start),
                    "end": int(end),
                    "probs": [float(prob)]
                }
        
        # Выводим общее количество I-ANN без B-ANN, а не каждый отдельно
        if i_ann_without_b_ann_count > 0:
            logger.warning(f"Фрагмент {chunk_idx+1}: Найдено {i_ann_without_b_ann_count} токенов I-ANN без предшествующего B-ANN")
        
        # Добавляем последний span, если он есть
        if current_span is not None:
            annotation_spans.append(current_span)
        
        # Если нет аннотаций, возвращаем None
        if not annotation_spans:
            return None
        
        # Выбираем лучший span (по длине и уверенности)
        try:
            best_span = max(annotation_spans, key=lambda x: len(x["probs"]) * sum(x["probs"]) / len(x["probs"]))
            
            # Вычисляем среднюю уверенность модели
            avg_confidence = sum(best_span["probs"]) / len(best_span["probs"])
            
            # Проверка границ текста
            if best_span["end"] > len(text):
                logger.warning(f"Фрагмент {chunk_idx+1}: Конец span-а ({best_span['end']}) выходит за границы текста ({len(text)})")
                best_span["end"] = len(text)
            
            extracted_text = text[best_span["start"]:best_span["end"]]
            
            return {
                "start": best_span["start"],
                "end": best_span["end"],
                "text": extracted_text,
                "confidence": float(avg_confidence)
            }
        except Exception as e:
            logger.error(f"Фрагмент {chunk_idx+1}: Ошибка при выборе лучшего span-а: {e}")
            return None

def predict_annotation(
    text: Optional[str] = None,
    file_path: Optional[str] = PREDICT_FILE_PATH,
    output_path: Optional[str] = PREDICT_OUTPUT_PATH,
    model_dir: str = OUTPUT_DIR
) -> str:
    """
    Предсказывает аннотацию для текста или файла
    
    Args:
        text: Текст для анализа (если None, будет использован файл)
        file_path: Путь к файлу с текстом (если None, будет использован text)
        output_path: Путь для сохранения результата (если None, результат будет возвращен)
        model_dir: Директория с моделью
    
    Returns:
        Строка предсказанной аннотации
    """
    # Проверяем входные данные
    if text is None and file_path is None:
        raise ValueError("Необходимо указать text или file_path")
    
    # Читаем текст из файла, если он не указан
    if text is None:
        logger.info(f"Чтение текста из файла {file_path}")
        
        # Анализируем файл предсказания для отладки
        diagnose_predict_file(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                logger.info(f"Прочитано {len(text)} символов из файла")
        except FileNotFoundError:
            logger.error(f"Файл не найден: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {e}")
            raise
    
    # Обрезаем текст до MAX_TEXT_LENGTH_FOR_PREDICTION символов
    if len(text) > MAX_TEXT_LENGTH_FOR_PREDICTION:
        logger.info(f"Обрезаем текст до {MAX_TEXT_LENGTH_FOR_PREDICTION} символов (исходная длина: {len(text)})")
        text = text[:MAX_TEXT_LENGTH_FOR_PREDICTION]
    
    # Создаем предсказатель
    try:
        predictor = AnnotationPredictor(model_dir)
        
        # Получаем предсказание
        annotation, fragments, confidence = predictor.predict(text)
        
        logger.info(f"Найдено {len(fragments)} фрагментов аннотации")
        logger.info(f"Уверенность в предсказании: {confidence:.4f}")
        
        # Формируем результат
        result = {
            "annotation": annotation,
            "confidence": confidence,
            "fragments": fragments
        }
        
        # Сохраняем результат в файл или возвращаем
        if output_path:
            logger.info(f"Сохранение результата в {output_path}")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"Результат сохранен в {output_path}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении результата: {e}")
        
        # Дополнительная отладка: если найдена аннотация, проверяем её содержимое
        if annotation:
            # Проверяем примерную длину аннотации для отладки
            words = annotation.split()
            logger.info(f"Найденная аннотация содержит {len(words)} слов и {len(annotation)} символов")
        else:
            logger.warning("Аннотация не найдена в тексте")
        
        return annotation
    
    except Exception as e:
        logger.error(f"Ошибка при предсказании аннотации: {e}")
        raise

if __name__ == "__main__":
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Убираем запись в файл, оставляем только вывод в консоль
            logging.StreamHandler()
        ]
    )
    
    logger.info("=" * 50)
    logger.info("Запуск предсказания аннотации")
    
    # Запускаем предсказание
    try:
        annotation = predict_annotation()
        print(f"\nПредсказанная аннотация:\n{annotation}")
        logger.info("Предсказание завершено успешно")
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказания: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")