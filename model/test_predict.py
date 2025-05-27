"""
Скрипт для тестирования предсказаний модели на примерах из test.json
"""

import os
import json
import logging
import torch
from typing import Dict, List, Any, Optional
from sklearn.metrics import classification_report
from pathlib import Path

from model.config import OUTPUT_DIR, MAX_TEXT_LENGTH_FOR_PREDICTION, ANNOTATION_LABELS
from model.predict import AnnotationPredictor

logger = logging.getLogger(__name__)

def load_test_examples(test_json_path: str) -> List[Dict[str, Any]]:
    """
    Загружает примеры из test.json
    
    Args:
        test_json_path: Путь к файлу test.json
        
    Returns:
        Список примеров
    """
    try:
        with open(test_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Загружено {len(data)} примеров из {test_json_path}")
        return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла {test_json_path}: {e}")
        return []

def test_predictions(examples: List[Dict[str, Any]], model_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    """
    Тестирует предсказания модели на примерах из test.json
    
    Args:
        examples: Список примеров из test.json
        model_dir: Директория с моделью
        
    Returns:
        Словарь с результатами тестирования
    """
    results = {
        "total_examples": len(examples),
        "successful_predictions": 0,
        "failed_predictions": 0,
        "correct_annotations": 0,
        "incorrect_annotations": 0,
        "examples": []
    }
    
    try:
        # Инициализируем предсказатель
        predictor = AnnotationPredictor(model_dir)
        logger.info(f"Предсказатель инициализирован из {model_dir}")
        
        # Обрабатываем каждый пример
        for i, example in enumerate(examples):
            logger.info(f"Обработка примера {i+1}/{len(examples)}")
            
            try:
                # Получаем текст и эталонную аннотацию
                text = example["text"]
                true_annotation = example["annotation"]
                true_start = example.get("start_idx", -1)
                true_end = example.get("end_idx", -1)
                
                # Обрезаем текст до MAX_TEXT_LENGTH_FOR_PREDICTION символов
                original_length = len(text)
                if len(text) > MAX_TEXT_LENGTH_FOR_PREDICTION:
                    logger.info(f"Пример {i+1}: Обрезаем текст до {MAX_TEXT_LENGTH_FOR_PREDICTION} символов (исходная длина: {len(text)})")
                    text = text[:MAX_TEXT_LENGTH_FOR_PREDICTION]
                    
                    # Обновляем true_start и true_end, если они находятся внутри обрезанного текста
                    if 0 <= true_start < MAX_TEXT_LENGTH_FOR_PREDICTION and 0 <= true_end <= MAX_TEXT_LENGTH_FOR_PREDICTION:
                        logger.info(f"Пример {i+1}: Эталонная аннотация находится внутри обрезанного текста")
                    else:
                        logger.warning(f"Пример {i+1}: Эталонная аннотация находится за пределами обрезанного текста")
                
                # Делаем предсказание
                predicted_annotation, fragments, confidence = predictor.predict(text)
                
                # Собираем информацию о предсказании
                example_result = {
                    "id": i,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "original_length": original_length,
                    "processed_length": len(text),
                    "true_annotation": true_annotation,
                    "true_start": true_start,
                    "true_end": true_end,
                    "predicted_annotation": predicted_annotation,
                    "confidence": float(confidence),
                    "fragments": fragments,
                    "is_correct": predicted_annotation.strip() == true_annotation.strip()
                }
                
                # Обновляем статистику
                results["successful_predictions"] += 1
                if example_result["is_correct"]:
                    results["correct_annotations"] += 1
                else:
                    results["incorrect_annotations"] += 1
                
                results["examples"].append(example_result)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке примера {i+1}: {e}")
                results["failed_predictions"] += 1
                results["examples"].append({
                    "id": i,
                    "error": str(e),
                    "text_preview": text[:100] + "..." if len(text) > 100 else text
                })
        
        # Вычисляем общую точность
        if results["successful_predictions"] > 0:
            results["accuracy"] = results["correct_annotations"] / results["successful_predictions"]
        else:
            results["accuracy"] = 0.0
            
        # Добавляем анализ фрагментов
        fragment_analysis = analyze_fragments(results)
        results["fragment_analysis"] = fragment_analysis
        
        # Добавляем анализ меток для токенов (для B-ANN, I-ANN и O)
        token_metrics = analyze_token_metrics(predictor, examples)
        results["token_metrics"] = token_metrics
        
        # Добавляем подробные ошибки по каждому примеру
        for i, example in enumerate(results["examples"]):
            if "per_example_errors" in token_metrics and i < len(token_metrics["per_example_errors"]):
                example.update(token_metrics["per_example_errors"][i])
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании предсказаний: {e}")
        return results

def analyze_token_metrics(predictor: AnnotationPredictor, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Анализирует метрики для каждой метки (B-ANN, I-ANN, O)
    Возвращает также подробные ошибки по каждому примеру.
    """
    all_true_labels = []
    all_pred_labels = []
    per_example_errors = []

    for example in examples:
        text = example["text"]
        if len(text) > MAX_TEXT_LENGTH_FOR_PREDICTION:
            text = text[:MAX_TEXT_LENGTH_FOR_PREDICTION]
        true_start = example.get("start_idx", -1)
        true_end = example.get("end_idx", -1)
        encodings = predictor.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            stride=0,
            max_length=512,
            padding="max_length",
            truncation=True
        )
        offset_mapping = encodings["offset_mapping"][0].numpy()
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        device = predictor.model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = predictor.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]
        true_labels = [ANNOTATION_LABELS["O"]] * len(offset_mapping)
        if 0 <= true_start < MAX_TEXT_LENGTH_FOR_PREDICTION and 0 <= true_end <= MAX_TEXT_LENGTH_FOR_PREDICTION:
            first = True
            for i, (start, end) in enumerate(offset_mapping):
                if end == 0:
                    continue
                if start >= true_start and end <= true_end:
                    if first:
                        true_labels[i] = ANNOTATION_LABELS["B-ANN"]
                        first = False
                    else:
                        true_labels[i] = ANNOTATION_LABELS["I-ANN"]
        valid_indices = [i for i, mask in enumerate(attention_mask[0].cpu().numpy()) if mask == 1]
        label_map = {v: k for k, v in ANNOTATION_LABELS.items()}
        errors = []
        for i in valid_indices:
            if offset_mapping[i][1] > 0:
                all_true_labels.append(true_labels[i])
                all_pred_labels.append(predictions[i])
                if true_labels[i] != predictions[i]:
                    errors.append({
                        "position": int(offset_mapping[i][0]),
                        "true": label_map[true_labels[i]],
                        "pred": label_map[predictions[i]]
                    })
        per_example_errors.append({
            "errors_count": len(errors),
            "errors": errors,
            "true_labels": sum(1 for i in valid_indices if offset_mapping[i][1] > 0),
            "pred_labels": sum(1 for i in valid_indices if offset_mapping[i][1] > 0)
        })
    all_true_labels_str = [label_map[lbl] for lbl in all_true_labels]
    all_pred_labels_str = [label_map[lbl] for lbl in all_pred_labels]
    report = classification_report(all_true_labels_str, all_pred_labels_str, output_dict=True)
    error_types = {
        "false_positive_B": 0,
        "false_positive_I": 0,
        "false_negative_B": 0,
        "false_negative_I": 0,
        "confusion_B_I": 0,
    }
    for true, pred in zip(all_true_labels_str, all_pred_labels_str):
        if true == "O" and pred == "B-ANN":
            error_types["false_positive_B"] += 1
        elif true == "O" and pred == "I-ANN":
            error_types["false_positive_I"] += 1
        elif true == "B-ANN" and pred == "O":
            error_types["false_negative_B"] += 1
        elif true == "I-ANN" and pred == "O":
            error_types["false_negative_I"] += 1
        elif (true == "B-ANN" and pred == "I-ANN") or (true == "I-ANN" and pred == "B-ANN"):
            error_types["confusion_B_I"] += 1
    return {
        "classification_report": report,
        "error_types": error_types,
        "confusion_matrix": {
            "total_tokens": len(all_true_labels),
            "correct_tokens": sum(1 for t, p in zip(all_true_labels_str, all_pred_labels_str) if t == p)
        },
        "per_example_errors": per_example_errors
    }

def analyze_fragments(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Анализирует фрагменты и причины ошибок
    
    Args:
        results: Результаты тестирования
        
    Returns:
        Словарь с анализом фрагментов
    """
    analysis = {
        "fragment_stats": {
            "total_fragments": 0,
            "avg_fragments_per_example": 0,
            "avg_confidence": 0,
            "start_position_error": 0,
            "end_position_error": 0
        },
        "error_analysis": {
            "no_fragments_found": 0,
            "wrong_fragment_selected": 0,
            "boundary_errors": 0
        }
    }
    
    try:
        # Собираем статистику по фрагментам
        total_fragments = 0
        total_confidence = 0
        start_position_diff = 0
        end_position_diff = 0
        examples_with_fragments = 0
        
        for example in results["examples"]:
            if "fragments" not in example or not example["fragments"]:
                analysis["error_analysis"]["no_fragments_found"] += 1
                continue
                
            fragments = example["fragments"]
            total_fragments += len(fragments)
            examples_with_fragments += 1
            
            # Вычисляем среднюю уверенность
            fragment_confidences = [f["confidence"] for f in fragments]
            total_confidence += sum(fragment_confidences) / len(fragment_confidences)
            
            # Если есть информация о позиции эталонной аннотации
            if example.get("true_start", -1) >= 0 and example.get("true_end", -1) >= 0:
                # Находим ближайший фрагмент к эталонной аннотации
                best_fragment = None
                min_distance = float('inf')
                
                for fragment in fragments:
                    start_diff = abs(fragment["start"] - example["true_start"])
                    end_diff = abs(fragment["end"] - example["true_end"])
                    distance = start_diff + end_diff
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_fragment = fragment
                
                if best_fragment:
                    start_position_diff += abs(best_fragment["start"] - example["true_start"])
                    end_position_diff += abs(best_fragment["end"] - example["true_end"])
                    
                    # Анализ ошибок границ
                    if abs(best_fragment["start"] - example["true_start"]) > 20 or \
                       abs(best_fragment["end"] - example["true_end"]) > 20:
                        analysis["error_analysis"]["boundary_errors"] += 1
        
        # Вычисляем средние значения
        if examples_with_fragments > 0:
            analysis["fragment_stats"]["total_fragments"] = total_fragments
            analysis["fragment_stats"]["avg_fragments_per_example"] = total_fragments / examples_with_fragments
            analysis["fragment_stats"]["avg_confidence"] = total_confidence / examples_with_fragments
            analysis["fragment_stats"]["start_position_error"] = start_position_diff / examples_with_fragments
            analysis["fragment_stats"]["end_position_error"] = end_position_diff / examples_with_fragments
        
        # Анализируем случаи, когда был выбран неправильный фрагмент
        for example in results["examples"]:
            if not example.get("is_correct", True) and "fragments" in example and len(example["fragments"]) > 1:
                # Если есть несколько фрагментов, но выбран неправильный
                analysis["error_analysis"]["wrong_fragment_selected"] += 1
                
        return analysis
        
    except Exception as e:
        logger.error(f"Ошибка при анализе фрагментов: {e}")
        return analysis

def print_test_results(results: Dict[str, Any]) -> None:
    """
    Выводит результаты тестирования в консоль
    
    Args:
        results: Результаты тестирования
    """
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ НА ПРИМЕРАХ")
    print("=" * 80)
    
    print(f"\nВсего примеров: {results['total_examples']}")
    print(f"Успешных предсказаний: {results['successful_predictions']}")
    print(f"Неудачных предсказаний: {results['failed_predictions']}")
    print(f"Правильных аннотаций: {results['correct_annotations']}")
    print(f"Неправильных аннотаций: {results['incorrect_annotations']}")
    print(f"Точность: {results.get('accuracy', 0):.4f}")
    
    # Вывод статистики фрагментов
    if "fragment_analysis" in results:
        frag_analysis = results["fragment_analysis"]
        
        print("\n--- Анализ фрагментов ---")
        frag_stats = frag_analysis["fragment_stats"]
        print(f"Среднее количество фрагментов на пример: {frag_stats['avg_fragments_per_example']:.2f}")
        print(f"Средняя уверенность: {frag_stats['avg_confidence']:.4f}")
        print(f"Средняя ошибка начальной позиции: {frag_stats['start_position_error']:.2f} символов")
        print(f"Средняя ошибка конечной позиции: {frag_stats['end_position_error']:.2f} символов")
        
        print("\n--- Анализ ошибок фрагментов ---")
        err_analysis = frag_analysis["error_analysis"]
        print(f"Не найдено фрагментов: {err_analysis['no_fragments_found']}")
        print(f"Выбран неправильный фрагмент: {err_analysis['wrong_fragment_selected']}")
        print(f"Ошибки границ: {err_analysis['boundary_errors']}")
    
    # Вывод отчета о классификации
    if "token_metrics" in results and "classification_report" in results["token_metrics"]:
        report = results["token_metrics"]["classification_report"]
        
        print("\n--- Отчет о классификации токенов ---")
        print("-" * 60)
        print(f"{'Метка':15} {'Precision':10} {'Recall':10} {'F1-score':10} {'Support':10}")
        print("-" * 60)
        
        for label in sorted(report.keys()):
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics = report[label]
                print(f"{label:15} {metrics['precision']:.4f}{'':<6} {metrics['recall']:.4f}{'':<6} {metrics['f1-score']:.4f}{'':<6} {metrics['support']:<10}")
        
        print("-" * 60)
        print(f"{'Accuracy':15} {'':<10} {'':<10} {report['accuracy']:.4f}{'':<6} {sum([report[l]['support'] for l in report if l not in ['accuracy', 'macro avg', 'weighted avg']])}")
        print("-" * 60)
        print(f"{'Macro avg':15} {report['macro avg']['precision']:.4f}{'':<6} {report['macro avg']['recall']:.4f}{'':<6} {report['macro avg']['f1-score']:.4f}")
        print(f"{'Weighted avg':15} {report['weighted avg']['precision']:.4f}{'':<6} {report['weighted avg']['recall']:.4f}{'':<6} {report['weighted avg']['f1-score']:.4f}")
    
    # Вывод типов ошибок меток
    if "token_metrics" in results and "error_types" in results["token_metrics"]:
        errors = results["token_metrics"]["error_types"]
        
        print("\n--- Анализ типов ошибок меток ---")
        for error_type, count in errors.items():
            print(f"{error_type}: {count}")
    
    print("=" * 80)

def run_test(model_dir: str = OUTPUT_DIR, test_json_path: Optional[str] = None, output_path: Optional[str] = None):
    """
    Запускает тестирование модели на примерах из test.json
    
    Args:
        model_dir: Директория с моделью
        test_json_path: Путь к файлу test.json (если None, используется стандартный)
        output_path: Путь для сохранения результатов (если None, используется стандартный)
    """
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Определяем пути к файлам
    if test_json_path is None:
        test_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "prepare-dataset-for-training", "dataset", "test.json")
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "test_results.json")
    
    logger.info(f"Тестирование предсказаний на примерах из {test_json_path}")
    
    # Загружаем примеры
    examples = load_test_examples(test_json_path)
    
    if not examples:
        logger.error("Не удалось загрузить тестовые примеры. Завершение работы.")
        return
    
    # Запускаем тестирование
    results = test_predictions(examples, model_dir)
    
    # Выводим результаты
    print_test_results(results)
    
    # Сохраняем результаты в один файл
    Path(os.path.dirname(output_path)).mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Все результаты сохранены в {output_path}")

if __name__ == "__main__":
    run_test()