"""
Вспомогательные функции для проекта извлечения текста из PDF
"""

from pathlib import Path


def create_directory_structure(pdf_path, output_dir):
    """
    Создает структуру директорий для сохранения результатов
    
    Args:
        pdf_path: путь к исходному PDF файлу
        output_dir: корневая директория для сохранения результатов
        
    Returns:
        dict: словарь с путями к директориям
    """
    pdf_name = Path(pdf_path).stem
    base_dir = Path(output_dir) / pdf_name
    images_dir = base_dir / "images"
    texts_dir = base_dir / "texts"
    
    # Создаем директории, если они не существуют
    images_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "base_dir": base_dir,
        "images_dir": images_dir,
        "texts_dir": texts_dir,
        "raw_text_path": texts_dir / "raw_text.txt",
        "processed_text_path": texts_dir / "processed_text.txt"
    }


def save_text_to_file(text, file_path):
    """
    Сохраняет текст в файл
    
    Args:
        text: текст для сохранения
        file_path: путь к файлу
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text) 