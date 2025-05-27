"""
Модуль для распознавания текста из PDF-файлов с использованием OCR
"""

import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
from tqdm import tqdm

from image_processor import ImagePreprocessor


class OCREngine:
    """Класс для распознавания текста из PDF"""
    
    def __init__(self, tesseract_path=None, poppler_path=None, lang='rus+eng',
                 preprocessing_enabled=False, preprocess_params=None):
        """
        Инициализация OCR-движка
        
        Args:
            tesseract_path: путь к исполняемому файлу Tesseract
            poppler_path: путь к директории с Poppler
            lang: языки для распознавания
            preprocessing_enabled: включить предобработку изображений
            preprocess_params: параметры предобработки (словарь)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        self.poppler_path = poppler_path
        self.lang = lang
        self.preprocessing_enabled = preprocessing_enabled
        
        # Создаем экземпляр предобработчика изображений
        if preprocessing_enabled:
            self.preprocessor = ImagePreprocessor(
                denoise=preprocess_params.get('denoise', True),
                contrast_enhance=preprocess_params.get('contrast_enhance', True)
            )
        else:
            self.preprocessor = None
    
    def process_pdf(self, pdf_path, output_paths, dpi=300, save_images=False):
        """
        Обработка PDF-файла и извлечение текста
        
        Args:
            pdf_path: путь к PDF-файлу
            output_paths: словарь с путями для сохранения результатов
            dpi: разрешение изображений
            save_images: сохранять обработанные изображения
            
        Returns:
            str: распознанный текст
        """
        pdf_path = Path(pdf_path)
        
        print(f"Конвертация PDF в изображения: {pdf_path}")
        try:
            images = convert_from_path(
                pdf_path, 
                dpi=dpi,
                poppler_path=self.poppler_path
            )
        except Exception as e:
            print(f"Ошибка при конвертации PDF {pdf_path}: {e}")
            return None
            
        print(f"Распознавание текста из {len(images)} страниц...")
        full_text = ""
        
        for i, image in enumerate(tqdm(images, desc="Обработка страниц")):
            # Предобработка изображения, если включена
            if self.preprocessing_enabled and self.preprocessor:
                processed_image = self.preprocessor.process_image(image)
                
                # Сохраняем обработанное изображение, если нужно
                if save_images:
                    self.preprocessor.save_image(
                        processed_image, 
                        output_paths['images_dir'], 
                        i + 1
                    )
            else:
                processed_image = image
                
                # Сохраняем оригинальное изображение, если нужно
                if save_images:
                    image_path = Path(output_paths['images_dir']) / f"page_{i+1:03d}.png"
                    image.save(str(image_path))
                
            # Распознавание текста
            page_text = pytesseract.image_to_string(processed_image, lang=self.lang)
            
            # Добавляем текст в общий результат
            full_text += f"\n\n--- Страница {i+1} ---\n\n"
            full_text += page_text
            
        return full_text 