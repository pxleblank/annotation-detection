"""
Модуль для обработки PDF-файлов и координации работы компонентов
"""

from pathlib import Path
from tqdm import tqdm

from ocr_engine import OCREngine
from text_processor import TextProcessor
from utils import create_directory_structure, save_text_to_file


class PDFProcessor:
    """Класс для обработки PDF-файлов"""
    
    def __init__(self, config):
        """
        Инициализация обработчика PDF
        
        Args:
            config: конфигурационные настройки
        """
        self.config = config
        
        # Инициализация OCR-движка
        self.ocr = OCREngine(
            tesseract_path=config.TESSERACT_PATH,
            poppler_path=config.POPPLER_PATH,
            lang=config.OCR_LANG,
            preprocessing_enabled=config.PREPROCESSING_ENABLED,
            preprocess_params={
                'denoise': config.DENOISE,
                'contrast_enhance': config.CONTRAST_ENHANCE
            }
        )
        
        # Инициализация обработчика текста
        self.text_processor = TextProcessor(
            use_nltk=config.USE_NLTK,
            use_spacy=config.USE_SPACY,
            use_symspell=config.USE_SYMSPELL,
            symspell_dict_path=config.SYMSPELL_DICT
        )
    
    def process_pdf(self, pdf_path):
        """
        Обработка PDF-файла
        
        Args:
            pdf_path: путь к PDF-файлу
            
        Returns:
            dict: информация о результатах обработки
        """
        # Создаем структуру директорий для результатов
        paths = create_directory_structure(pdf_path, self.config.OUTPUT_DIR)
        
        # Извлечение текста из PDF с помощью OCR
        raw_text = self.ocr.process_pdf(
            pdf_path, 
            paths, 
            dpi=self.config.OCR_DPI,
            save_images=self.config.SAVE_IMAGES
        )
        
        if raw_text is None:
            print(f"Не удалось обработать файл {pdf_path}")
            return None
        
        # Сохраняем сырой текст
        save_text_to_file(raw_text, paths['raw_text_path'])
        
        # Применяем постобработку текста, если включена
        if self.config.POSTPROCESSING_ENABLED:
            processed_text = self.text_processor.process_text(raw_text)
            # Сохраняем обработанный текст
            save_text_to_file(processed_text, paths['processed_text_path'])
        else:
            processed_text = raw_text
            
        return {
            'raw_text': raw_text,
            'processed_text': processed_text,
            'paths': paths
        }
        
    def process_directory(self, input_dir=None):
        """
        Обработка всех PDF-файлов в директории
        
        Args:
            input_dir: директория с PDF-файлами (если None, используется из конфигурации)
            
        Returns:
            list: список результатов обработки
        """
        if input_dir is None:
            input_dir = self.config.INPUT_DIR
            
        input_dir = Path(input_dir)
        
        # Находим все PDF-файлы в указанной директории
        pdf_files = list(input_dir.glob('*.pdf'))
        print(f"Найдено {len(pdf_files)} PDF файлов в {input_dir}")
        
        results = []
        
        for pdf_file in tqdm(pdf_files, desc="Обработка файлов"):
            print(f"\nОбработка {pdf_file.name}")
            result = self.process_pdf(pdf_file)
            if result:
                results.append(result)
                
        return results 