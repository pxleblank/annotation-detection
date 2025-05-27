#!/usr/bin/env python3
"""
Скрипт для постобработки готового текста без выполнения OCR
"""

import sys
from pathlib import Path

from text_processor import TextProcessor

# Настройки по умолчанию
# Пути к файлам
INPUT_PATH = r"C:\PyProjects\annotation-detection\extract-text-from-pdf\raw_texts"  # Путь к входному файлу или директории
OUTPUT_PATH = r"C:\PyProjects\annotation-detection\extract-text-from-pdf\postprocessed_texts2"  # Путь к выходному файлу или директории

# Настройки обработки
USE_NLTK = False  # Использовать NLTK для обработки
USE_SPACY = True  # Использовать SpaCy для обработки
USE_SYMSPELL = True  # Использовать SymSpell для исправления опечаток
SYMSPELL_DICT = "ru-100k.txt"  # Путь к словарю SymSpell


def process_file(input_file, output_file, use_nltk=USE_NLTK, use_spacy=USE_SPACY, use_symspell=USE_SYMSPELL, 
               symspell_dict=SYMSPELL_DICT):
    """
    Обрабатывает отдельный текстовый файл
    
    Args:
        input_file: путь к входному файлу
        output_file: путь к выходному файлу
        use_nltk: использовать NLTK
        use_spacy: использовать SpaCy
        use_symspell: использовать SymSpell
        symspell_dict: путь к словарю SymSpell
    """
    print(f"Обрабатываю файл: {input_file}")
    
    # Инициализируем процессор текста
    processor = TextProcessor(
        use_nltk=use_nltk,
        use_spacy=use_spacy,
        use_symspell=use_symspell,
        symspell_dict_path=symspell_dict
    )
    
    # Читаем входной файл
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Обрабатываем текст
    processed_text = processor.process_text(text)
    
    # Сохраняем обработанный текст
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    print(f"Обработанный текст сохранен в: {output_file}")


def process_directory(input_dir, output_dir, use_nltk=USE_NLTK, use_spacy=USE_SPACY, use_symspell=USE_SYMSPELL, 
                    symspell_dict=SYMSPELL_DICT):
    """
    Обрабатывает все .txt файлы в директории
    
    Args:
        input_dir: путь к входной директории
        output_dir: путь к выходной директории
        use_nltk: использовать NLTK
        use_spacy: использовать SpaCy
        use_symspell: использовать SymSpell
        symspell_dict: путь к словарю SymSpell
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Находим все текстовые файлы
    txt_files = list(input_dir.glob('**/*.txt'))
    
    if not txt_files:
        print(f"В директории {input_dir} не найдено текстовых файлов")
        return
    
    print(f"Найдено {len(txt_files)} текстовых файлов")
    
    # Инициализируем процессор текста
    processor = TextProcessor(
        use_nltk=use_nltk,
        use_spacy=use_spacy,
        use_symspell=use_symspell,
        symspell_dict_path=symspell_dict
    )
    
    for txt_file in txt_files:
        # Определяем путь к выходному файлу
        rel_path = txt_file.relative_to(input_dir)
        out_file = output_dir / rel_path
        
        # Создаем директории, если нужно
        out_file.parent.mkdir(exist_ok=True, parents=True)
        
        print(f"Обрабатываю: {txt_file}")
        
        # Читаем входной файл
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Обрабатываем текст
        processed_text = processor.process_text(text)
        
        # Сохраняем обработанный текст
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        
        print(f"Сохранено в: {out_file}")


def main():
    """
    Основная функция скрипта
    """
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    
    if input_path.is_file():
        # Обработка одного файла
        process_file(
            input_path, 
            output_path, 
            use_nltk=USE_NLTK,
            use_spacy=USE_SPACY,
            use_symspell=USE_SYMSPELL,
            symspell_dict=SYMSPELL_DICT
        )
    elif input_path.is_dir():
        # Обработка директории
        process_directory(
            input_path, 
            output_path, 
            use_nltk=USE_NLTK,
            use_spacy=USE_SPACY,
            use_symspell=USE_SYMSPELL,
            symspell_dict=SYMSPELL_DICT
        )
    else:
        print(f"Ошибка: {input_path} не существует")
        sys.exit(1)


if __name__ == "__main__":
    main() 