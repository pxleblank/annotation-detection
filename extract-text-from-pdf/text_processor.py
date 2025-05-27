"""
Модуль для постобработки распознанного текста
"""

import re
from pathlib import Path

class TextProcessor:
    """Класс для постобработки распознанного текста"""
    
    def __init__(self, use_nltk=False, use_spacy=False, use_symspell=False, 
                 symspell_dict_path=None):
        """
        Инициализация обработчика текста
        
        Args:
            use_nltk: использовать NLTK для обработки текста
            use_spacy: использовать SpaCy для обработки текста
            use_symspell: использовать SymSpell для исправления опечаток
            symspell_dict_path: путь к словарю SymSpell
        """
        self.use_nltk = use_nltk
        self.use_spacy = use_spacy
        self.use_symspell = use_symspell
        
        # Инициализируем компоненты для обработки текста
        self.nltk_processor = NLTKProcessor() if use_nltk else None
        self.spacy_processor = SpacyProcessor() if use_spacy else None
        self.symspell_processor = None
        
        # Инициализируем SymSpell, если указан
        if use_symspell and symspell_dict_path:
            self.symspell_processor = SymSpellProcessor(symspell_dict_path)
    
    def process_text(self, text):
        """
        Обрабатывает текст с использованием выбранных методов
        
        Args:
            text: распознанный текст для обработки
            
        Returns:
            str: обработанный текст
        """
        processed_text = text
        
        # Базовая очистка текста (удаление лишних пробелов, переносов строк)
        processed_text = self._basic_cleanup(processed_text)
        
        # Исправление опечаток с помощью SymSpell
        if self.use_symspell and self.symspell_processor:
            processed_text = self.symspell_processor.correct_text(processed_text)
            
        # Обработка с помощью NLTK
        if self.use_nltk and self.nltk_processor:
            processed_text = self.nltk_processor.process(processed_text)
            
        # Обработка с помощью SpaCy
        if self.use_spacy and self.spacy_processor:
            processed_text = self.spacy_processor.process(processed_text)

        # Склеиваем переносы слов типа "био- логический" -> "биологический"
        processed_text = re.sub(r'([а-яА-ЯёЁ]+)-\s+([а-яА-ЯёЁ]+)', r'\1\2', processed_text)
            
        return processed_text
    
    def _basic_cleanup(self, text):
        """
        Базовая очистка текста
        
        Args:
            text: исходный текст
            
        Returns:
            str: очищенный текст
        """
        # Удаляем двойные пробелы
        text = re.sub(r'\s+', ' ', text)
        # Удаляем пробелы в начале и конце строк
        text = '\n'.join(line.strip() for line in text.split('\n'))
        # Удаляем пустые строки
        text = re.sub(r'\n+', '\n', text)
        return text


class NLTKProcessor:
    """Класс для обработки текста с помощью NLTK"""
    
    # Флаг для отслеживания попытки загрузки ресурса
    _downloaded_punkt_tab = False
    
    def __init__(self):
        """Инициализация NLTK-процессора"""
        try:
            import nltk
            
            # Проверяем наличие punkt_tab и загружаем только если нужно
            if not NLTKProcessor._downloaded_punkt_tab:
                print("Загрузка punkt_tab NLTK...")
                nltk.download('punkt_tab')
                NLTKProcessor._downloaded_punkt_tab = True
                
            # Проверяем, установлены ли необходимые ресурсы
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Загрузка пунктуации NLTK...")
                nltk.download('punkt')
                
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("Загрузка стоп-слов NLTK...")
                nltk.download('stopwords')
        except ImportError:
            print("NLTK не установлен. Выполните: pip install nltk")
            self.nltk_available = False
        else:
            self.nltk_available = True
    
    def process(self, text):
        """
        Обработка текста с помощью NLTK
        
        Args:
            text: исходный текст
            
        Returns:
            str: обработанный текст
        """
        if not self.nltk_available:
            return text
            
        import nltk
        if not NLTKProcessor._downloaded_punkt_tab:
                print("Загрузка punkt_tab NLTK...")
                nltk.download('punkt_tab')
                NLTKProcessor._downloaded_punkt_tab = True
        from nltk.tokenize import word_tokenize, sent_tokenize
        
        # Токенизация по предложениям
        sentences = sent_tokenize(text, language='russian')
        
        # Обработка каждого предложения
        processed_sentences = []
        for sentence in sentences:
            # Токенизация по словам
            words = word_tokenize(sentence, language='russian')
            
            # Собираем предложение обратно
            processed_sentence = ' '.join(words)
            processed_sentences.append(processed_sentence)
            
        # Собираем текст обратно
        processed_text = ' '.join(processed_sentences)
        
        return processed_text


class SpacyProcessor:
    """Класс для обработки текста с помощью SpaCy"""
    
    def __init__(self):
        """Инициализация SpaCy-процессора"""
        try:
            import spacy
            try:
                self.nlp = spacy.load('ru_core_news_sm')
            except:
                print("Загрузка модели SpaCy для русского языка...")
                spacy.cli.download('ru_core_news_sm')
                self.nlp = spacy.load('ru_core_news_sm')
        except ImportError:
            print("SpaCy не установлен. Выполните: pip install spacy")
            self.spacy_available = False
        else:
            self.spacy_available = True
    
    def process(self, text):
        """
        Обработка текста с помощью SpaCy
        
        Args:
            text: исходный текст
            
        Returns:
            str: обработанный текст
        """
        if not self.spacy_available:
            return text
            
        # Обработка текста с помощью SpaCy
        doc = self.nlp(text)
        
        # Извлекаем предложения и восстанавливаем правильную пунктуацию
        processed_sentences = []
        for sent in doc.sents:
            processed_sentences.append(sent.text)
            
        # Собираем текст обратно
        processed_text = ' '.join(processed_sentences)
        
        return processed_text


class SymSpellProcessor:
    """Класс для исправления опечаток с помощью SymSpell"""
    
    def __init__(self, dictionary_path):
        """
        Инициализация SymSpell-процессора
        
        Args:
            dictionary_path: путь к словарю
        """
        try:
            from symspellpy import SymSpell, Verbosity
            
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            
            # Загружаем словарь
            dictionary_path = Path(dictionary_path)
            if dictionary_path.exists():
                self.sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
                self.symspell_available = True
            else:
                print(f"Словарь SymSpell не найден: {dictionary_path}")
                self.symspell_available = False
                
            self.verbosity = Verbosity.CLOSEST
                
        except ImportError:
            print("SymSpellPy не установлен. Выполните: pip install symspellpy")
            self.symspell_available = False
    
    def correct_text(self, text):
        """
        Исправление опечаток в тексте
        
        Args:
            text: исходный текст
            
        Returns:
            str: текст с исправленными опечатками
        """
        if not self.symspell_available:
            return text
            
        from symspellpy import Verbosity
        
        # Разбиваем текст на предложения и обрабатываем каждое отдельно
        lines = text.split('\n')
        corrected_lines = []
        
        for line in lines:
            # Пропускаем пустые строки и маркеры страниц
            if not line.strip() or line.strip().startswith('---'):
                corrected_lines.append(line)
                continue
                
            # Исправляем опечатки
            suggestions = self.sym_spell.lookup_compound(line, max_edit_distance=2)
            if suggestions:
                corrected_lines.append(suggestions[0].term)
            else:
                corrected_lines.append(line)
                
        return '\n'.join(corrected_lines)