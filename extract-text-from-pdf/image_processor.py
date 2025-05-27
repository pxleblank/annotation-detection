"""
Модуль для предобработки изображений перед OCR
"""

import cv2
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    """Класс для предобработки изображений перед OCR"""
    
    def __init__(self, denoise=True, contrast_enhance=True):
        """
        Инициализация предобработчика изображений
        
        Args:
            denoise: применять удаление шумов
            contrast_enhance: применять улучшение контраста
        """
        self.denoise = denoise
        self.contrast_enhance = contrast_enhance
        
    def process_image(self, image):
        """
        Применяет методы предобработки к изображению
        
        Args:
            image: PIL-изображение для обработки
            
        Returns:
            PIL-изображение после обработки
        """
        # Конвертируем PIL Image в формат, подходящий для OpenCV
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Конвертируем в градации серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Применяем удаление шумов
        if self.denoise:
            gray = self._apply_denoise(gray)
            
        # Улучшаем контрастность
        if self.contrast_enhance:
            gray = self._apply_contrast_enhancement(gray)
        
        # Преобразуем обратно в PIL Image
        processed_image = image.copy()
        processed_image.paste(Image.fromarray(gray), (0, 0))
        
        return processed_image
    
    def _apply_denoise(self, img):
        """Применяет фильтр для удаления шумов"""
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    
    def _apply_contrast_enhancement(self, img):
        """Улучшает контрастность изображения"""
        # Применяем CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    def save_image(self, image, path, page_num):
        """
        Сохраняет обработанное изображение
        
        Args:
            image: PIL-изображение для сохранения
            path: директория для сохранения
            page_num: номер страницы
            
        Returns:
            str: путь к сохраненному файлу
        """
        image_path = Path(path) / f"page_{page_num:03d}.png"
        image.save(str(image_path))
        return str(image_path)


# Импортируем PIL только после определения класса,
# чтобы избежать циклических импортов
from PIL import Image 