import argparse
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm

class PDFTextExtractor:
    def __init__(self, tesseract_path=None, poppler_path=None, lang='rus+eng'):
        """
        Инициализация экстрактора текста из PDF
        
        Args:
            tesseract_path (str): Путь к исполняемому файлу Tesseract OCR
            poppler_path (str): Путь к папке с исполняемыми файлами Poppler
            lang (str): Языки для распознавания (rus+eng по умолчанию)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.poppler_path = poppler_path
        self.lang = lang
    
    def extract_text_from_pdf(self, pdf_path, output_dir=None, dpi=300):
        """
        Извлечение текста из PDF файла
        
        Args:
            pdf_path (str): Путь к PDF файлу
            output_dir (str): Директория для сохранения результатов
            dpi (int): Разрешение изображений в DPI
            
        Returns:
            str: Распознанный текст
        """
        pdf_path = Path(pdf_path)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"{pdf_path.stem}.txt"
        
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
            page_text = pytesseract.image_to_string(image, lang=self.lang)
            full_text += f"\n\n--- Страница {i+1} ---\n\n"
            full_text += page_text
            
        if output_dir:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"Текст сохранен в {output_file}")
            
        return full_text
    
    def process_directory(self, input_dir, output_dir, dpi=300):
        """
        Обработка всех PDF файлов в директории
        
        Args:
            input_dir (str): Директория с PDF файлами
            output_dir (str): Директория для сохранения результатов
            dpi (int): Разрешение изображений в DPI
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        pdf_files = list(input_dir.glob('*.pdf'))
        print(f"Найдено {len(pdf_files)} PDF файлов в {input_dir}")
        
        for pdf_file in tqdm(pdf_files, desc="Обработка файлов"):
            print(f"\nОбработка {pdf_file.name}")
            self.extract_text_from_pdf(pdf_file, output_dir, dpi)

def main():
    parser = argparse.ArgumentParser(description="Извлечение текста из PDF с помощью OCR")
    parser.add_argument("--input", "-i", required=True, help="Путь к PDF файлу или директории с PDF файлами")
    parser.add_argument("--output", "-o", required=True, help="Путь для сохранения извлеченного текста")
    parser.add_argument("--tesseract", help="Путь к исполняемому файлу Tesseract")
    parser.add_argument("--poppler", help="Путь к директории с исполняемыми файлами Poppler")
    parser.add_argument("--lang", default="rus+eng", help="Языки для распознавания (по умолчанию: rus+eng)")
    parser.add_argument("--dpi", type=int, default=300, help="Разрешение изображений в DPI (по умолчанию: 300)")
    
    args = parser.parse_args()
    
    extractor = PDFTextExtractor(
        tesseract_path=args.tesseract,
        poppler_path=args.poppler,
        lang=args.lang
    )
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        extractor.process_directory(input_path, args.output, args.dpi)
    else:
        extractor.extract_text_from_pdf(input_path, args.output, args.dpi)

if __name__ == "__main__":
    main() 