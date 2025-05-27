#!/usr/bin/env python3
"""
Основная точка входа для проекта извлечения текста из PDF
"""

import time
from pathlib import Path

import config
from pdf_processor import PDFProcessor


def main():
    """
    Основная функция для запуска обработки PDF-файлов
    """
    start_time = time.time()
    
    print("Инициализация обработчика PDF...")
    processor = PDFProcessor(config)
    
    # Если указан путь к отдельному файлу, обрабатываем только его
    if Path(config.INPUT_DIR).is_file() and Path(config.INPUT_DIR).suffix.lower() == '.pdf':
        print(f"Обработка файла: {config.INPUT_DIR}")
        processor.process_pdf(config.INPUT_DIR)
    else:
        # Иначе обрабатываем все PDF-файлы в директории
        processor.process_directory()
    
    elapsed_time = time.time() - start_time
    print(f"Обработка завершена за {elapsed_time:.2f} секунд")


if __name__ == "__main__":
    main() 