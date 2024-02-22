"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
"""

import json
import logging
import sys

from bot import Bot

from dialogues import Dialogues


def logging_setup() -> None:
    """Настройка формата и уровня ведения журнала"""

    # Создание форматтера журнала
    log_formatter = logging.Formatter(
        "%(asctime)s %(threadName)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Настройка вывода журнала в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # Добавление всех обработчиков и установка уровня
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    # Журналирование тестового сообщения
    logging.info("Настройка журнала завершена")


def main() -> None:
    """Основной вход"""
    # Инициализация журнала
    logging_setup()

    # Загрузка сообщений из JSON
    logging.info("Загрузка сообщений из bot_config.json")
    with open("bot_config.json", "r", encoding="utf-8") as file:
        bot_config = json.load(file)

    # Инициализация обработчика диалогов
    dialogue = Dialogues(bot_config)

    # Инициализация класса бота
    bot_handler = Bot(dialogue)

    # Разбор диалогов
    dialogue.parse_dialogues_from_file()

    # Обучение
    dialogue.train_classifier()

    # Запуск бота
    bot_handler.start()


if __name__ == "__main__":
    main()
