#!/usr/bin/env python3
"""
Графический интерфейс для обработки PDF-файлов и постобработки текста
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any
import yaml
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
    QSpinBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QGridLayout, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Импортируем модули нашего проекта
import config
from pdf_processor import PDFProcessor
from text_processor import TextProcessor


class WorkerThread(QThread):
    """Поток для выполнения длительных операций"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, target_func, args):
        super().__init__()
        self.target_func = target_func
        self.args = args

    def run(self):
        try:
            self.target_func(*self.args)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class ProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF и текст: обработка и постобработка")
        self.setGeometry(100, 100, 800, 700)

        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Создаем вкладки
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Создаем вкладки для PDF и текста
        self.pdf_tab = QWidget()
        self.text_tab = QWidget()
        self.tabs.addTab(self.pdf_tab, "Обработка PDF")
        self.tabs.addTab(self.text_tab, "Постобработка текста")

        # Настраиваем вкладки
        self._setup_pdf_tab()
        self._setup_text_tab()

        # Лог выполнения
        log_group = QGroupBox("Лог выполнения")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Статус бар
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")

    def _setup_pdf_tab(self):
        """Настройка вкладки для обработки PDF"""
        layout = QVBoxLayout(self.pdf_tab)

        # Группа путей
        paths_group = QGroupBox("Пути")
        paths_layout = QGridLayout(paths_group)

        # Входной путь
        paths_layout.addWidget(QLabel("Входной PDF или директория:"), 0, 0)
        self.pdf_input_edit = QLineEdit(config.INPUT_DIR)
        paths_layout.addWidget(self.pdf_input_edit, 0, 1)
        pdf_input_btn = QPushButton("Обзор")
        pdf_input_btn.clicked.connect(self._browse_pdf_input)
        paths_layout.addWidget(pdf_input_btn, 0, 2)

        # Выходной путь
        paths_layout.addWidget(QLabel("Выходная директория:"), 1, 0)
        self.pdf_output_edit = QLineEdit(config.OUTPUT_DIR)
        paths_layout.addWidget(self.pdf_output_edit, 1, 1)
        pdf_output_btn = QPushButton("Обзор")
        pdf_output_btn.clicked.connect(self._browse_pdf_output)
        paths_layout.addWidget(pdf_output_btn, 1, 2)

        layout.addWidget(paths_group)

        # Группа OCR
        ocr_group = QGroupBox("Настройки OCR")
        ocr_layout = QGridLayout(ocr_group)

        # DPI
        ocr_layout.addWidget(QLabel("DPI:"), 0, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(config.OCR_DPI)
        ocr_layout.addWidget(self.dpi_spin, 0, 1)

        # Языки
        ocr_layout.addWidget(QLabel("Языки:"), 1, 0)
        self.lang_edit = QLineEdit(config.OCR_LANG)
        ocr_layout.addWidget(self.lang_edit, 1, 1)

        layout.addWidget(ocr_group)

        # Группа предобработки
        preproc_group = QGroupBox("Предобработка изображений")
        preproc_layout = QGridLayout(preproc_group)

        self.preproc_check = QCheckBox("Включить предобработку")
        self.preproc_check.setChecked(config.PREPROCESSING_ENABLED)
        preproc_layout.addWidget(self.preproc_check, 0, 0)

        self.denoise_check = QCheckBox("Удаление шумов")
        self.denoise_check.setChecked(config.DENOISE)
        preproc_layout.addWidget(self.denoise_check, 1, 0)

        self.contrast_check = QCheckBox("Улучшение контраста")
        self.contrast_check.setChecked(config.CONTRAST_ENHANCE)
        preproc_layout.addWidget(self.contrast_check, 1, 1)

        layout.addWidget(preproc_group)

        # Группа постобработки
        postproc_group = QGroupBox("Постобработка текста")
        postproc_layout = QGridLayout(postproc_group)

        self.postproc_check = QCheckBox("Включить постобработку")
        self.postproc_check.setChecked(config.POSTPROCESSING_ENABLED)
        postproc_layout.addWidget(self.postproc_check, 0, 0)

        self.nltk_check = QCheckBox("NLTK")
        self.nltk_check.setChecked(config.USE_NLTK)
        postproc_layout.addWidget(self.nltk_check, 1, 0)

        self.spacy_check = QCheckBox("SpaCy")
        self.spacy_check.setChecked(config.USE_SPACY)
        postproc_layout.addWidget(self.spacy_check, 1, 1)

        self.symspell_check = QCheckBox("SymSpell")
        self.symspell_check.setChecked(config.USE_SYMSPELL)
        postproc_layout.addWidget(self.symspell_check, 1, 2)

        # Словарь SymSpell
        postproc_layout.addWidget(QLabel("Словарь SymSpell:"), 2, 0)
        self.pdf_symspell_dict_edit = QLineEdit(config.SYMSPELL_DICT)
        postproc_layout.addWidget(self.pdf_symspell_dict_edit, 2, 1, 1, 2)
        pdf_symspell_dict_btn = QPushButton("Обзор")
        pdf_symspell_dict_btn.clicked.connect(self._browse_pdf_symspell_dict)
        postproc_layout.addWidget(pdf_symspell_dict_btn, 2, 3)

        layout.addWidget(postproc_group)

        # Дополнительные настройки
        other_group = QGroupBox("Дополнительно")
        other_layout = QVBoxLayout(other_group)

        self.save_images_check = QCheckBox("Сохранять изображения")
        self.save_images_check.setChecked(config.SAVE_IMAGES)
        other_layout.addWidget(self.save_images_check)

        layout.addWidget(other_group)

        # Кнопка запуска
        self.process_pdf_btn = QPushButton("Начать обработку PDF")
        self.process_pdf_btn.clicked.connect(self._start_pdf_processing)
        layout.addWidget(self.process_pdf_btn)

    def _setup_text_tab(self):
        """Настройка вкладки для постобработки текста"""
        layout = QVBoxLayout(self.text_tab)

        # Группа путей
        paths_group = QGroupBox("Пути")
        paths_layout = QGridLayout(paths_group)

        # Входной путь
        paths_layout.addWidget(QLabel("Входной текст или директория:"), 0, 0)
        self.text_input_edit = QLineEdit()
        paths_layout.addWidget(self.text_input_edit, 0, 1)
        text_input_btn = QPushButton("Обзор")
        text_input_btn.clicked.connect(self._browse_text_input)
        paths_layout.addWidget(text_input_btn, 0, 2)

        # Выходной путь
        paths_layout.addWidget(QLabel("Выходной файл или директория:"), 1, 0)
        self.text_output_edit = QLineEdit()
        paths_layout.addWidget(self.text_output_edit, 1, 1)
        text_output_btn = QPushButton("Обзор")
        text_output_btn.clicked.connect(self._browse_text_output)
        paths_layout.addWidget(text_output_btn, 1, 2)

        layout.addWidget(paths_group)

        # Группа обработки
        proc_group = QGroupBox("Настройки обработки")
        proc_layout = QGridLayout(proc_group)

        self.text_nltk_check = QCheckBox("NLTK")
        self.text_nltk_check.setChecked(True)
        proc_layout.addWidget(self.text_nltk_check, 0, 0)

        self.text_spacy_check = QCheckBox("SpaCy")
        self.text_spacy_check.setChecked(True)
        proc_layout.addWidget(self.text_spacy_check, 0, 1)

        self.text_symspell_check = QCheckBox("SymSpell")
        self.text_symspell_check.setChecked(True)
        proc_layout.addWidget(self.text_symspell_check, 0, 2)

        # Словарь SymSpell
        proc_layout.addWidget(QLabel("Словарь SymSpell:"), 1, 0)
        self.symspell_dict_edit = QLineEdit(config.SYMSPELL_DICT)
        proc_layout.addWidget(self.symspell_dict_edit, 1, 1, 1, 2)
        symspell_dict_btn = QPushButton("Обзор")
        symspell_dict_btn.clicked.connect(self._browse_symspell_dict)
        proc_layout.addWidget(symspell_dict_btn, 1, 3)

        layout.addWidget(proc_group)

        # Кнопка запуска
        self.process_text_btn = QPushButton("Начать обработку текста")
        self.process_text_btn.clicked.connect(self._start_text_processing)
        layout.addWidget(self.process_text_btn)

    def _browse_pdf_input(self):
        """Выбор входного PDF-файла или директории"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите PDF файл", "",
            "PDF файлы (*.pdf);;Все файлы (*.*)"
        )
        if not path:  # Пользователь отменил выбор файла
            path = QFileDialog.getExistingDirectory(
                self, "Выберите директорию с PDF файлами"
            )
        if path:
            self.pdf_input_edit.setText(path)

    def _browse_pdf_output(self):
        """Выбор выходной директории для PDF"""
        path = QFileDialog.getExistingDirectory(
            self, "Выберите выходную директорию"
        )
        if path:
            self.pdf_output_edit.setText(path)

    def _browse_text_input(self):
        """Выбор входного текстового файла или директории"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите текстовый файл", "",
            "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if not path:  # Пользователь отменил выбор файла
            path = QFileDialog.getExistingDirectory(
                self, "Выберите директорию с текстовыми файлами"
            )
        if path:
            self.text_input_edit.setText(path)

    def _browse_text_output(self):
        """Выбор выходного файла или директории для текста"""
        input_path = self.text_input_edit.text()

        if input_path and Path(input_path).is_file():
            path, _ = QFileDialog.getSaveFileName(
                self, "Выберите файл для сохранения", "",
                "Текстовые файлы (*.txt);;Все файлы (*.*)"
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Выберите выходную директорию"
            )

        if path:
            self.text_output_edit.setText(path)

    def _browse_symspell_dict(self):
        """Выбор словаря SymSpell"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите словарь SymSpell", "",
            "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if path:
            self.symspell_dict_edit.setText(path)

    def _browse_pdf_symspell_dict(self):
        """Выбор словаря SymSpell для вкладки обработки PDF"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите словарь SymSpell", "",
            "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if path:
            self.pdf_symspell_dict_edit.setText(path)

    def _log(self, message):
        """Добавить сообщение в лог"""
        self.log_text.append(message)

    def _update_status(self, message):
        """Обновить статус"""
        self.status_bar.showMessage(message)

    def _update_progress(self, value):
        """Обновить прогресс-бар"""
        self.progress_bar.setValue(value)

    def _start_pdf_processing(self):
        """Начать обработку PDF"""
        # Проверяем пути
        input_path = self.pdf_input_edit.text()
        output_path = self.pdf_output_edit.text()

        if not input_path or not output_path:
            QMessageBox.critical(self, "Ошибка", "Необходимо указать входной и выходной пути")
            return

        # Отключаем кнопку на время выполнения
        self.process_pdf_btn.setEnabled(False)

        # Создаем временную конфигурацию
        temp_config = type('Config', (), {})()
        temp_config.TESSERACT_PATH = config.TESSERACT_PATH
        temp_config.POPPLER_PATH = config.POPPLER_PATH
        temp_config.SYMSPELL_DICT = self.pdf_symspell_dict_edit.text()

        temp_config.OCR_LANG = self.lang_edit.text()
        temp_config.OCR_DPI = self.dpi_spin.value()

        temp_config.PREPROCESSING_ENABLED = self.preproc_check.isChecked()
        temp_config.DENOISE = self.denoise_check.isChecked()
        temp_config.CONTRAST_ENHANCE = self.contrast_check.isChecked()

        temp_config.POSTPROCESSING_ENABLED = self.postproc_check.isChecked()
        temp_config.USE_NLTK = self.nltk_check.isChecked()
        temp_config.USE_SPACY = self.spacy_check.isChecked()
        temp_config.USE_SYMSPELL = self.symspell_check.isChecked()

        temp_config.INPUT_DIR = input_path
        temp_config.OUTPUT_DIR = output_path

        temp_config.SAVE_IMAGES = self.save_images_check.isChecked()

        # Создаем и запускаем поток
        self.worker = WorkerThread(self._process_pdf, (temp_config,))
        self.worker.progress.connect(self._update_progress)
        self.worker.status.connect(self._update_status)
        self.worker.log.connect(self._log)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(lambda: self.process_pdf_btn.setEnabled(True))
        self.worker.start()

    def _process_pdf(self, config):
        """Обработка PDF в отдельном потоке"""
        try:
            self.worker.log.emit("Начинаем обработку PDF...")
            self.worker.status.emit("Обработка PDF...")
            self.worker.progress.emit(0)

            processor = PDFProcessor(config)

            input_path = Path(config.INPUT_DIR)

            if input_path.is_file():
                self.worker.log.emit(f"Обрабатываем файл: {input_path}")
                result = processor.process_pdf(input_path)
                self.worker.log.emit(f"Обработка завершена. Результат сохранен в: {result['paths']['texts_dir']}")
            else:
                self.worker.log.emit(f"Обрабатываем директорию: {input_path}")
                results = processor.process_directory()
                self.worker.log.emit(f"Обработка завершена. Обработано файлов: {len(results)}")

            self.worker.status.emit("Обработка PDF завершена")
            self.worker.progress.emit(100)

        except Exception as e:
            self.worker.error.emit(str(e))

    def _start_text_processing(self):
        """Начать постобработку текста"""
        # Проверяем пути
        input_path = self.text_input_edit.text()
        output_path = self.text_output_edit.text()

        if not input_path or not output_path:
            QMessageBox.critical(self, "Ошибка", "Необходимо указать входной и выходной пути")
            return

        # Отключаем кнопку на время выполнения
        self.process_text_btn.setEnabled(False)

        # Получаем параметры
        use_nltk = self.text_nltk_check.isChecked()
        use_spacy = self.text_spacy_check.isChecked()
        use_symspell = self.text_symspell_check.isChecked()
        symspell_dict = self.symspell_dict_edit.text()

        # Создаем и запускаем поток
        self.worker = WorkerThread(
            self._process_text,
            (input_path, output_path, use_nltk, use_spacy, use_symspell, symspell_dict)
        )
        self.worker.progress.connect(self._update_progress)
        self.worker.status.connect(self._update_status)
        self.worker.log.connect(self._log)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(lambda: self.process_text_btn.setEnabled(True))
        self.worker.start()

    def _process_text(self, input_path, output_path, use_nltk, use_spacy, use_symspell, symspell_dict):
        """Постобработка текста в отдельном потоке"""
        try:
            self.worker.log.emit("Начинаем обработку текста...")
            self.worker.status.emit("Обработка текста...")
            self.worker.progress.emit(0)

            input_path = Path(input_path)

            # Создаем экземпляр обработчика текста
            processor = TextProcessor(
                use_nltk=use_nltk,
                use_spacy=use_spacy,
                use_symspell=use_symspell,
                symspell_dict_path=symspell_dict
            )

            if input_path.is_file():
                self.worker.log.emit(f"Обрабатываем файл: {input_path}")

                # Читаем входной файл
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Обрабатываем текст
                processed_text = processor.process_text(text)

                # Сохраняем обработанный текст
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processed_text)

                self.worker.log.emit(f"Обработка завершена. Результат сохранен в: {output_path}")
            else:
                self.worker.log.emit(f"Обрабатываем директорию: {input_path}")
                output_dir = Path(output_path)
                output_dir.mkdir(exist_ok=True, parents=True)

                # Находим все текстовые файлы
                txt_files = list(input_path.glob('**/*.txt'))
                total_files = len(txt_files)

                if not txt_files:
                    self.worker.log.emit(f"В директории {input_path} не найдено текстовых файлов")
                    self.worker.status.emit("Нет файлов для обработки")
                    self.worker.progress.emit(100)
                    return

                self.worker.log.emit(f"Найдено {total_files} текстовых файлов")

                # Обрабатываем каждый файл
                for i, txt_file in enumerate(txt_files):
                    # Определяем путь к выходному файлу
                    rel_path = txt_file.relative_to(input_path)
                    out_file = output_dir / rel_path

                    # Создаем директории, если нужно
                    out_file.parent.mkdir(exist_ok=True, parents=True)

                    self.worker.log.emit(f"Обрабатываю [{i+1}/{total_files}]: {txt_file}")

                    # Читаем входной файл
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read()

                    # Обрабатываем текст
                    processed_text = processor.process_text(text)

                    # Сохраняем обработанный текст
                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write(processed_text)

                    # Обновляем прогресс
                    progress = ((i + 1) / total_files) * 100
                    self.worker.progress.emit(progress)

                self.worker.log.emit(f"Обработка завершена. Результаты сохранены в: {output_dir}")

            self.worker.status.emit("Обработка текста завершена")
            self.worker.progress.emit(100)

        except Exception as e:
            self.worker.error.emit(str(e))

    def _handle_error(self, error_message):
        """Обработка ошибок"""
        self._log(f"Ошибка: {error_message}")
        self._update_status(f"Ошибка: {error_message}")
        QMessageBox.critical(self, "Ошибка", str(error_message))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProcessorGUI()
    window.show()
    sys.exit(app.exec_())