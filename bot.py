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

import asyncio
import logging

from telegram import (
    Update,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

from dialogues import Dialogues

# Токен бота
BOT_TOKEN = ""


class Bot:
    def __init__(self, dialogues: Dialogues) -> None:
        self._dialogues = dialogues

        self._event_loop = None
        self._application = None

    def start(self) -> None:
        """Создает бота и запускает опрос"""
        # Закрытие предыдущего цикла событий (на всякий случай)
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                logging.info("Остановка текущего цикла событий перед запуском нового")
                loop.stop()
        except Exception as e:
            logging.warning(f"Ошибка остановки текущего цикла событий: {str(e)}")

        # Создание нового цикла событий
        logging.info("Создание нового цикла событий")
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        # Создание бота
        builder = ApplicationBuilder().token(BOT_TOKEN)
        self._application = builder.build()

        # Чтение команды /start
        self._application.add_handler(CommandHandler("start", self._bot_callback_start))

        # Чтение сообщений
        self._application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self._bot_callback_message))

        # Запуск опроса бота
        logging.info("Запуск опроса бота. Нажмите CTRL+C для остановки")
        self._application.run_polling(close_loop=True)

    async def _bot_callback_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обрабатывает команду /start"""
        # Получение ID пользователя
        chat_id = update.effective_chat.id

        # Отправка стартового сообщения
        responses = self._dialogues.next_message("start", chat_id)
        for response in responses:
            await context.bot.send_message(chat_id=chat_id, text=response)

    async def _bot_callback_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обрабатывает сообщения"""
        # Получение ID пользователя
        chat_id = update.effective_chat.id

        # Извлечение текстового запроса
        if update.message.caption:
            request_message = update.message.caption.strip()
        elif context.args is not None:
            request_message = str(" ".join(context.args)).strip()
        elif update.message.text:
            request_message = update.message.text.strip()
        else:
            request_message = ""

        if not request_message:
            return

        # Отправка следующего сообщения
        responses = self._dialogues.next_message(request_message, chat_id)
        for response in responses:
            await context.bot.send_message(chat_id=chat_id, text=response)
