# Инструкция по работе с репозиторием 

## 1. Клонирование репозитория

```bash
git clone <URL-репозитория>
```

## 2. Создание виртуального окружения

Рекомендуется создавать отдельное виртуальное окружение для работы с заданиями на Python. Например, в корне репозитория:

```bash
python -m venv venv
```

- Папка `venv` будет содержать все зависимости и настройки Python.
- Добавьте `venv` в `.gitignore`, чтобы не попадала в репозиторий.

## 3. Активация виртуального окружения

- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (cmd):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

## 4. Установка зависимостей

Если есть файл `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 5. Запуск Python-скриптов

Переходите в нужную папку (например, LAB3) и запускайте скрипты:

```bash
python main.py
```

## 6. Работа с Jupyter Notebook

- Установите Jupyter:
  ```bash
  pip install notebook
  ```
- Запустите:
  ```bash
  jupyter notebook
  ```
- Откройте файл, например, `LAB1/lab1.ipynb`.

## 7. Настройка PyCharm

- Откройте репозиторий в PyCharm.
- Перейдите в "Settings" → "Project: ComputerGraphics" → "Python Interpreter".
- Выберите или добавьте виртуальное окружение из папки `venv`.
- Убедитесь, что `.idea` и `venv` добавлены в `.gitignore`.

## 8. Общие рекомендации

- Не добавляйте служебные папки IDE (`.idea`, `.vscode`) и виртуальное окружение (`venv`) в репозиторий.
- Для новых лабораторных работ создавайте отдельные папки.
- Все зависимости фиксируйте в `requirements.txt`.

## 9. Обновление зависимостей

Если появились новые зависимости:

```bash
pip freeze > requirements.txt
```

## 10. Полезные команды

- Проверить установленные пакеты:
  ```bash
  pip list
  ```
- Деактивировать виртуальное окружение:
  ```bash
  deactivate
  ```

---
