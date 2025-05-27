"""
Конфигурационный файл для проекта извлечения текста из PDF
"""

# Пути к внешним инструментам
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Путь к исполняемому файлу Tesseract
POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"  # Путь к директории с исполняемыми файлами Poppler
SYMSPELL_DICT = r"C:\PyProjects\annotation-detection\extract-text-from-pdf\ru-100k.txt"  # Путь к словарю для SymSpell

# Настройки OCR
OCR_LANG = "rus+eng"  # Языки для распознавания
OCR_DPI = 300  # Разрешение изображений в DPI

# Настройки предобработки изображений
PREPROCESSING_ENABLED = True  # Включить предобработку изображений
DENOISE = True  # Удаление шумов
CONTRAST_ENHANCE = True  # Улучшение контрастности

# Настройки постобработки текста
POSTPROCESSING_ENABLED = True  # Включить постобработку текста
USE_NLTK = False  # Использовать NLTK для постобработки
USE_SPACY = True  # Использовать SpaCy для постобработки
USE_SYMSPELL = True  # Использовать SymSpell для исправления опечаток

# Пути к директориям
INPUT_DIR = r"C:\PyProjects\annotation-detection\extract-text-from-pdf\Для ИИ\Добавленное"  # Директория с PDF файлами
OUTPUT_DIR = r"C:\PyProjects\annotation-detection\extract-text-from-pdf\extracted-text\addition"  # Корневая директория для результатов

# Настройки сохранения
SAVE_IMAGES = False  # Сохранять обработанные изображения