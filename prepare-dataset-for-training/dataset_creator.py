import os
import json
from typing import Dict, List
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QLabel, QWidget, 
                           QMessageBox, QFrame, QGroupBox, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor
import random
import sys

class AnnotationDatasetCreator:
    def __init__(self, output_dir: str = "dataset"):
        """
        Инициализирует создатель датасета для обучения модели выделения аннотаций.
        
        Args:
            output_dir: Директория для сохранения датасета
        """
        self.output_dir = output_dir
        self.dataset = []
        
        # Создаем директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Пути к файлам датасета
        self.train_path = os.path.join(output_dir, "train.json")
        self.val_path = os.path.join(output_dir, "val.json")
        self.test_path = os.path.join(output_dir, "test.json")
        
        # Загружаем существующие данные, если они есть
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Загружает существующие данные из файлов, если они существуют."""
        all_data = []
        for path in [self.train_path, self.val_path, self.test_path]:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_data.extend(data)
                except Exception as e:
                    print(f"Ошибка при загрузке данных из {path}: {e}")
        
        self.dataset = all_data
    
    def add_example(self, text: str, annotation: str) -> Dict:
        """
        Добавляет пример в датасет.
        
        Args:
            text: Полный текст документа
            annotation: Текст аннотации
            
        Returns:
            Dict: Созданный пример
        """
        if not text or not annotation:
            raise ValueError("Текст и аннотация не могут быть пустыми")
        
        if annotation not in text:
            raise ValueError("Аннотация должна быть частью текста")
        
        # Найти индексы начала и конца аннотации в тексте
        start_idx = text.find(annotation)
        end_idx = start_idx + len(annotation)
        
        # Создаем пример
        example = {
            "text": text,
            "annotation": annotation,
            "start_idx": start_idx,
            "end_idx": end_idx
        }
        
        # Добавляем в датасет
        self.dataset.append(example)
        return example
    
    def save_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.15, test_ratio: float = 0.05, shuffle: bool = False):
        """
        Сохраняет датасет в файлы с разделением на обучающую, валидационную и тестовую выборки.
        
        Args:
            train_ratio: Доля примеров для обучающей выборки
            val_ratio: Доля примеров для валидационной выборки
            test_ratio: Доля примеров для тестовой выборки
            shuffle: Перемешивать ли датасет перед разделением
        """
        if not self.dataset:
            raise ValueError("Датасет пуст")
            
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
            raise ValueError("Сумма пропорций должна быть равна 1")
        
        # Создаем копию датасета
        dataset_copy = self.dataset.copy()
        
        # Перемешиваем датасет только если параметр shuffle=True
        if shuffle:
            random.shuffle(dataset_copy)
        
        # Разделяем на выборки
        n_samples = len(dataset_copy)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_data = dataset_copy[:train_size]
        val_data = dataset_copy[train_size:train_size+val_size]
        test_data = dataset_copy[train_size+val_size:]
        
        # Сохраняем в файлы
        self._save_to_file(self.train_path, train_data)
        self._save_to_file(self.val_path, val_data)
        self._save_to_file(self.test_path, test_data)
        
        print(f"Датасет сохранен: {len(train_data)} обучающих, {len(val_data)} валидационных, {len(test_data)} тестовых примеров")
    
    def _save_to_file(self, path: str, data: List[Dict]):
        """Сохраняет данные в JSON файл."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> Dict:
        """Возвращает статистику по датасету."""
        if not self.dataset:
            return {"total": 0}
        
        annotation_lengths = [len(example["annotation"]) for example in self.dataset]
        text_lengths = [len(example["text"]) for example in self.dataset]
        
        return {
            "total": len(self.dataset),
            "avg_annotation_length": sum(annotation_lengths) / len(annotation_lengths),
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "min_annotation_length": min(annotation_lengths),
            "max_annotation_length": max(annotation_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths)
        }


class DatasetCreatorGUI(QMainWindow):
    """Графический интерфейс для создания датасета на PyQt5."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Создание датасета для выделения аннотаций")
        self.setGeometry(100, 100, 900, 700)
        
        self.creator = AnnotationDatasetCreator(output_dir="dataset")
        self.examples_since_save = 0
        self.auto_save_threshold = 5  # Автосохранение после каждых 5 примеров
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Настраивает интерфейс."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Верхняя панель с информацией
        info_frame = QFrame()
        info_layout = QVBoxLayout(info_frame)
        
        stats = self.creator.get_stats()
        self.stats_label = QLabel(f"Всего примеров: {stats.get('total', 0)}")
        info_layout.addWidget(self.stats_label)
        
        main_layout.addWidget(info_frame)
        
        # Создаем разделитель для текста и аннотации
        splitter = QSplitter(Qt.Vertical)
        
        # Панель с текстом
        text_group = QGroupBox("Полный текст документа")
        text_layout = QVBoxLayout(text_group)
        
        self.text_widget = QTextEdit()
        self.text_widget.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_widget.setTabChangesFocus(True)
        self.text_widget.setAcceptRichText(False)
        text_layout.addWidget(self.text_widget)
        
        splitter.addWidget(text_group)
        
        # Панель с аннотацией
        annotation_group = QGroupBox("Аннотация")
        annotation_layout = QVBoxLayout(annotation_group)
        
        self.annotation_widget = QTextEdit()
        self.annotation_widget.setLineWrapMode(QTextEdit.WidgetWidth)
        self.annotation_widget.setTabChangesFocus(True)
        self.annotation_widget.setAcceptRichText(False)
        self.annotation_widget.setMaximumHeight(150)
        annotation_layout.addWidget(self.annotation_widget)
        
        splitter.addWidget(annotation_group)
        
        # Устанавливаем соотношение размеров
        splitter.setSizes([500, 200])
        
        main_layout.addWidget(splitter)
        
        # Панель с кнопками для вставки из буфера
        paste_layout = QHBoxLayout()
        
        self.paste_text_button = QPushButton("Вставить текст из буфера")
        self.paste_text_button.clicked.connect(self._paste_to_text)
        paste_layout.addWidget(self.paste_text_button)
        
        self.paste_annotation_button = QPushButton("Вставить аннотацию из буфера")
        self.paste_annotation_button.clicked.connect(self._paste_to_annotation)
        paste_layout.addWidget(self.paste_annotation_button)
        
        main_layout.addLayout(paste_layout)
        
        # Нижняя панель с кнопками
        buttons_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Добавить пример")
        self.add_button.clicked.connect(self._add_example)
        buttons_layout.addWidget(self.add_button)
        
        self.clear_button = QPushButton("Очистить поля")
        self.clear_button.clicked.connect(self._clear_fields)
        buttons_layout.addWidget(self.clear_button)
        
        # Добавляем растягивающийся пробел
        buttons_layout.addStretch()
        
        self.save_button = QPushButton("Сохранить датасет")
        self.save_button.clicked.connect(lambda: self._save_dataset(shuffle=False))
        buttons_layout.addWidget(self.save_button)
        
        self.save_shuffle_button = QPushButton("Сохранить с перемешиванием")
        self.save_shuffle_button.clicked.connect(lambda: self._save_dataset(shuffle=True))
        buttons_layout.addWidget(self.save_shuffle_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Устанавливаем горячие клавиши
        self._setup_shortcuts()
        
        # Обновить статистику
        self._update_stats()
    
    def _setup_shortcuts(self):
        """Настраивает горячие клавиши."""
        # Ctrl+S для сохранения
        self.save_shortcut = self.create_shortcut("Ctrl+S", lambda: self._save_dataset(shuffle=False))
        
        # Ctrl+A для выделения всего текста - обрабатывается нативно в PyQt5
    
    def create_shortcut(self, key_sequence, target):
        """Создает и возвращает горячую клавишу."""
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        shortcut = QShortcut(QKeySequence(key_sequence), self)
        shortcut.activated.connect(target)
        return shortcut
    
    def _paste_to_text(self):
        """Вставляет текст из буфера обмена в поле текста."""
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        
        # Проверяем длину текста
        if len(text) > 50000:
            reply = QMessageBox.question(self, "Предупреждение", 
                                        "Текст очень длинный (более 50000 символов), что может вызвать зависание. Продолжить?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        # Начинаем редактирование для корректной работы Undo
        self.text_widget.textCursor().beginEditBlock()
        
        # Выделяем весь текст
        cursor = self.text_widget.textCursor()
        cursor.select(QTextCursor.Document)
        self.text_widget.setTextCursor(cursor)
        
        # Вставляем новый текст
        self.text_widget.insertPlainText(text)
        
        # Завершаем редактирование
        self.text_widget.textCursor().endEditBlock()
    
    def _paste_to_annotation(self):
        """Вставляет текст из буфера обмена в поле аннотации."""
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        
        # Начинаем редактирование для корректной работы Undo
        self.annotation_widget.textCursor().beginEditBlock()
        
        # Выделяем весь текст
        cursor = self.annotation_widget.textCursor()
        cursor.select(QTextCursor.Document)
        self.annotation_widget.setTextCursor(cursor)
        
        # Вставляем новый текст
        self.annotation_widget.insertPlainText(text)
        
        # Завершаем редактирование
        self.annotation_widget.textCursor().endEditBlock()
    
    def _add_example(self):
        """Добавляет пример в датасет."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            text = self.text_widget.toPlainText().strip()
            annotation = self.annotation_widget.toPlainText().strip()
            
            self.creator.add_example(text, annotation)
            self._clear_fields()
            self._update_stats()
            
            self.examples_since_save += 1
            
            # Автосохранение после определенного количества примеров
            if self.examples_since_save >= self.auto_save_threshold:
                self._save_dataset(show_message=False, shuffle=False)
                self.examples_since_save = 0
            
            QMessageBox.information(self, "Успех", "Пример добавлен в датасет!")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            QApplication.restoreOverrideCursor()
    
    def _clear_fields(self):
        """Очищает поля ввода."""
        # Очистка поля текста с поддержкой отмены
        self.text_widget.textCursor().beginEditBlock()
        cursor = self.text_widget.textCursor()
        cursor.select(QTextCursor.Document)
        self.text_widget.setTextCursor(cursor)
        self.text_widget.insertPlainText("")
        self.text_widget.textCursor().endEditBlock()
        
        # Очистка поля аннотации с поддержкой отмены
        self.annotation_widget.textCursor().beginEditBlock()
        cursor = self.annotation_widget.textCursor()
        cursor.select(QTextCursor.Document)
        self.annotation_widget.setTextCursor(cursor)
        self.annotation_widget.insertPlainText("")
        self.annotation_widget.textCursor().endEditBlock()
    
    def _save_dataset(self, show_message=True, shuffle=False):
        """Сохраняет датасет."""
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            self.creator.save_dataset(shuffle=shuffle)
            self.examples_since_save = 0
            self._update_stats()
            
            QApplication.restoreOverrideCursor()
            
            if show_message:
                QMessageBox.information(self, "Успех", "Датасет сохранен!")
        except ValueError as e:
            QApplication.restoreOverrideCursor()
            if show_message:
                QMessageBox.critical(self, "Ошибка", str(e))
    
    def _update_stats(self):
        """Обновляет статистику."""
        stats = self.creator.get_stats()
        stats_text = f"Всего примеров: {stats.get('total', 0)}"
        
        if stats.get('total', 0) > 0:
            stats_text += f"\nСредняя длина текста: {stats.get('avg_text_length', 0):.1f} символов"
            stats_text += f"\nСредняя длина аннотации: {stats.get('avg_annotation_length', 0):.1f} символов"
        
        self.stats_label.setText(stats_text)


def main():
    app = QApplication(sys.argv)
    window = DatasetCreatorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 