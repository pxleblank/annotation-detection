"""
Pipeline для полного процесса обработки PDF и предсказания аннотации
Выполняет полный процесс от подачи PDF до предсказания аннотации без сохранения промежуточных результатов
"""

import os
import logging
from pathlib import Path
import tempfile
import sys
import importlib

# Добавляем путь к директории extract-text-from-pdf в sys.path,
# чтобы модули внутри нее могли импортировать друг друга
extract_text_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract-text-from-pdf")
sys.path.append(extract_text_dir)

# Импортируем модули с использованием importlib для избежания проблем с IDE
OCREngine = importlib.import_module("ocr_engine").OCREngine
TextProcessor = importlib.import_module("text_processor").TextProcessor

# Импортируем конфигурацию extract-text-from-pdf
pdf_config = importlib.import_module("config")

# Импорт компонентов для предсказания аннотации
from model.predict import AnnotationPredictor
# Импортируем конфигурацию модели
from model.config import MAX_TEXT_LENGTH_FOR_PREDICTION

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnnotationPipeline:
    """Класс для полного пайплайна обработки PDF и предсказания аннотации"""
    
    def __init__(self):
        """Инициализация пайплайна"""
        logger.info("Инициализация пайплайна для обработки PDF и предсказания аннотации")
        
        # Инициализация OCR-движка с использованием конфига из extract-text-from-pdf
        self.ocr = OCREngine(
            tesseract_path=pdf_config.TESSERACT_PATH,
            poppler_path=pdf_config.POPPLER_PATH,
            lang=pdf_config.OCR_LANG,
            preprocessing_enabled=pdf_config.PREPROCESSING_ENABLED,
            preprocess_params={
                'denoise': pdf_config.DENOISE,
                'contrast_enhance': pdf_config.CONTRAST_ENHANCE
            }
        )
        logger.info("OCR-движок инициализирован")
        
        # Инициализация обработчика текста с использованием конфига из extract-text-from-pdf
        self.text_processor = TextProcessor(
            use_nltk=pdf_config.USE_NLTK,
            use_spacy=pdf_config.USE_SPACY,
            use_symspell=pdf_config.USE_SYMSPELL,
            symspell_dict_path=pdf_config.SYMSPELL_DICT
        )
        logger.info("Обработчик текста инициализирован")
        
        # Инициализация предсказателя аннотаций с использованием конфига из model
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "models")
        self.predictor = AnnotationPredictor(model_dir)
        logger.info("Предсказатель аннотаций инициализирован")
    
    def process_pdf(self, pdf_path):
        """
        Обработка PDF-файла и предсказание аннотации
        
        Args:
            pdf_path: Путь к PDF-файлу
            
        Returns:
            dict: Результат обработки с предсказанной аннотацией
        """
        logger.info(f"Начало обработки PDF-файла: {pdf_path}")
        
        # Создаем временную директорию для промежуточных результатов
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            logger.info(f"Создана временная директория: {temp_dir_path}")
            
            # Структура путей для OCR
            paths = {
                'images_dir': temp_dir_path / 'images',
                'raw_text_path': temp_dir_path / 'raw_text.txt',
                'processed_text_path': temp_dir_path / 'processed_text.txt'
            }
            
            # Создаем директорию для изображений
            paths['images_dir'].mkdir(exist_ok=True)
            
            # Шаг 1: Извлечение текста из PDF с помощью OCR
            logger.info("Шаг 1: Извлечение текста из PDF")
            raw_text = self.ocr.process_pdf(
                pdf_path, 
                paths, 
                dpi=pdf_config.OCR_DPI,
                save_images=False
            )
            
            if raw_text is None:
                logger.error(f"Не удалось обработать файл {pdf_path}")
                return {"error": "Не удалось извлечь текст из PDF"}
            
            logger.info(f"Извлечено {len(raw_text)} символов текста")
            
            # Шаг 2: Постобработка текста
            logger.info("Шаг 2: Постобработка текста")
            if pdf_config.POSTPROCESSING_ENABLED:
                processed_text = self.text_processor.process_text(raw_text)
                logger.info(f"Текст обработан, получено {len(processed_text)} символов")
            else:
                processed_text = raw_text
                logger.info("Постобработка текста отключена")
            
            # Шаг 3: Предсказание аннотации
            logger.info("Шаг 3: Предсказание аннотации")
            
            # Обрезаем текст до MAX_TEXT_LENGTH_FOR_PREDICTION символов
            if len(processed_text) > MAX_TEXT_LENGTH_FOR_PREDICTION:
                logger.info(f"Обрезаем текст до {MAX_TEXT_LENGTH_FOR_PREDICTION} символов")
                processed_text = processed_text[:MAX_TEXT_LENGTH_FOR_PREDICTION]
            
            # Получаем предсказание
            annotation, fragments, confidence = self.predictor.predict(processed_text)
            
            logger.info(f"Предсказание завершено. Найдено {len(fragments)} фрагментов аннотации")
            logger.info(f"Уверенность в предсказании: {confidence:.4f}")
            
            # Формируем результат
            result = {
                "annotation": annotation,
                "confidence": confidence,
                "fragments": fragments
            }
            
            return result

def process_pdf_file(pdf_path):
    """
    Функция для обработки PDF-файла и получения аннотации
    
    Args:
        pdf_path: Путь к PDF-файлу
        
    Returns:
        str: Предсказанная аннотация
    """
    # Создаем пайплайн
    pipeline = AnnotationPipeline()
    
    # Обрабатываем PDF
    result = pipeline.process_pdf(pdf_path)
    
    # Возвращаем аннотацию
    return result.get("annotation", "")


if __name__ == "__main__":
    # Пример использования
    # Путь к PDF-файлу (замените на свой)
    pdf_path = r"C:\PyProjects\annotation-detection\test.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Файл не найден: {pdf_path}")
    else:
        try:
            # Создаем пайплайн и обрабатываем PDF
            pipeline = AnnotationPipeline()
            result = pipeline.process_pdf(pdf_path)
            
            # Выводим результат
            print("\n" + "=" * 50)
            print("РЕЗУЛЬТАТ ОБРАБОТКИ:")
            print("=" * 50)
            
            if "error" in result:
                print(f"Ошибка: {result['error']}")
            else:
                print(f"Предсказанная аннотация (уверенность: {result['confidence']:.4f}):")
                print("-" * 50)
                print(result["annotation"])
                print("-" * 50)
                
                print(f"\nНайдено {len(result['fragments'])} фрагментов аннотации")
                
        except Exception as e:
            logger.error(f"Ошибка при обработке файла: {e}", exc_info=True)
            print(f"Произошла ошибка: {e}") 