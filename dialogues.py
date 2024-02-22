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

import logging
import random
from typing import Dict, List

import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Вероятность рекламы
AD_PROBABILITY = 0.5

# Порог близости ответа из диалогов
DIALOGUES_THRESHOLD = 0.5

# Разрешённые символы
ALPHABET = "abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890- "


class Dialogues:
    def __init__(self, bot_config: Dict) -> None:
        self.bot_config = bot_config

        self._dialogues_structured = {}

        self._vectorizer = None
        self._classifier = None
        self._users_topics = {}

    def phrase_simplify(self, phrase: str) -> str:
        """Переводит фразу в нижний регистр и оставляет только символы ALPHABET

        Args:
            phrase (str): запрос на упрощение

        Returns:
            str: упрощенный запрос
        """
        return "".join(symbol for symbol in phrase.lower() if symbol in ALPHABET).strip()

    def parse_dialogues_from_file(self) -> None:
        """Загружает и парсит диалоги из dialogues.txt"""
        # Загрузить данные диалогов из файла
        logging.warning("Загрузка диалогов из dialogues.txt")
        with open("dialogues.txt", "r", encoding="utf-8") as file:
            content = file.read()

        # Разделить по двойным строкам
        dialogues = [dialogue.split("\n")[:2] for dialogue in content.split("\n\n") if len(dialogue.split("\n")) == 2]

        # Отфильтровать повторяющиеся вопросы и отформатировать фразы
        logging.info("Фильтрация диалогов")
        dialogues_filtered = []
        questions = set()
        for dialogue in dialogues:
            question, answer = dialogue
            question = self.phrase_simplify(question[2:])
            answer = answer[2:]
            if question and question not in questions:
                questions.add(question)
                dialogues_filtered.append([question, answer])

        # Создать словарь слов и их связанных пар вопрос-ответ
        logging.info("Структурирование диалогов")
        dialogues_structured = {}
        for question, answer in dialogues_filtered:
            words = set(question.split())
            for word in words:
                dialogues_structured.setdefault(word, []).append([question, answer])

        # Отсортировать пары по длине вопроса и оставить только первые 1000 для каждого слова
        logging.info("Сортировка диалогов")
        self._dialogues_structured = {
            word: sorted(pairs, key=lambda pair: len(pair[0]))[:1000] for word, pairs in dialogues_structured.items()
        }

        # Перемешать
        logging.info("Перемешивание диалогов")
        dialogues_structured_list = list(self._dialogues_structured.items())
        random.shuffle(dialogues_structured_list)
        self._dialogues_structured = dict(dialogues_structured_list)

        # Готово
        logging.info(f"Диалоги для {len(self._dialogues_structured)} слов загружены!")

    def train_classifier(self) -> None:
        """Создает и обучает классификатор намерений LinearSVC"""
        intent_names = []
        intent_examples = []

        # Обучение на dialogues.txt (только 5000 случайных намерений)
        for intent, dialogues_list in self._dialogues_structured.items():
            if len(intent_names) > 10000:
                break
            for dialogue_ in dialogues_list:
                intent_names.append(intent)
                intent_examples.append(dialogue_[0])

        # Обучение на обычных сообщениях
        for intent, intent_data in self.bot_config["intents"].items():
            for example in intent_data["examples"]:
                intent_names.append(intent)
                intent_examples.append(self.phrase_simplify(example))

        # Инициализация векторизатора
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))

        # Инициализация классификатора
        if self._classifier is None:
            self._classifier = LinearSVC(dual=True)

        # Обучение
        logging.info("Обучение классификатора намерений...")
        self._classifier.fit(self._vectorizer.fit_transform(intent_examples), intent_names)
        logging.info("Обучение завершено!")

    def intent_predict(self, request: str) -> str or None:
        """Пытается предсказать имя намерения на основе запроса

        Args:
            request (str): запрос для предсказания

        Returns:
            str or None: имя намерения или None в случае ошибки или не найдено
        """
        # Вернуть, если не обучен
        if self._vectorizer is None or self._classifier is None:
            return

        # Упростить запрос
        request = self.phrase_simplify(request)
        logging.info(f"Упрощенный запрос: {request}")

        # Предсказать первое намерение
        intent = self._classifier.predict(self._vectorizer.transform([request]))[0]
        logging.info(f'Предсказанное намерение для "{request}": {intent}')
        return intent

    def generate_answer_dialogues(self, replica) -> str or None:
        """Возвращает лучший ответ для данной реплики на основе dialogues.txt

        Args:
            replica (_type_): данная реплика

        Returns:
            str or None: лучший ответ или None, если не найден
        """
        replica = self.phrase_simplify(replica)
        words = set(replica.split())
        mini_dataset = [
            pair for word in words if word in self._dialogues_structured for pair in self._dialogues_structured[word]
        ]

        # [[distance_weighted, question, answer]]
        answers = []
        for question, answer in mini_dataset:
            if abs(len(replica) - len(question)) / len(question) < DIALOGUES_THRESHOLD:
                distance = nltk.edit_distance(replica, question)
                distance_weighted = distance / len(question)
                if distance_weighted < DIALOGUES_THRESHOLD:
                    answers.append([distance_weighted, question, answer])

        return min(answers, key=lambda three: three[0])[2] if answers else None

    def next_message(self, request: str, user_id: int) -> List[str]:
        """Генерирует список следующих сообщений из запроса и пользователя

        Args:
            request (str): новый запрос пользователя
            user_id (int): идентификатор пользователя (для хранения темы)

        Returns:
            List[str]: сообщения в виде строк
        """
        try:
            # Новый пользователь
            if not user_id in self._users_topics:
                logging.info(f"Создание нового пользователя: {user_id}")
                self._users_topics[user_id] = "any"

            # Извлечение темы
            user_topic = self._users_topics[user_id]

            # Предсказание намерения
            user_intent = self.intent_predict(request)

            # Лог запроса
            logging.info(
                f"Запрос от пользователя {user_id}: {request}. Тема пользователя: {user_topic}, намерение: {user_intent}"
            )

            # Проверка в обычных сообщениях
            if not user_intent or user_intent not in self.bot_config["intents"]:
                logging.warning(f"Намерение {user_intent} не найдено в сообщениях")
                # Использование dialogues.txt вместо
                answer_from_dialogues = self.generate_answer_dialogues(request)

                if answer_from_dialogues:
                    return [answer_from_dialogues]

                # Не в dialogues.txt
                else:
                    return [random.choice(self.bot_config["failure"])]

            # Проверка необходимости отправки рекламы
            ad = AD_PROBABILITY > random.random() and user_intent in self.bot_config["ad_intents"]
            if ad:
                logging.info("Реклама включена")

            # Выбор темы
            if user_topic in self.bot_config["intents"][user_intent]:
                topic = self.bot_config["intents"][user_intent][user_topic]
            else:
                topic = self.bot_config["intents"][user_intent]["any"]

            # Выбор ответа
            topic_response = random.choice(topic["responses"])

            # Получение следующей темы
            topic_next = topic["next"]

            # Установка следующей темы пользователя
            if ad:
                logging.info(f"Новая тема пользователя: {self.bot_config['ad_topic']}")
                self._users_topics[user_id] = self.bot_config["ad_topic"]
            elif topic_next != "keep":
                logging.info(f"Новая тема пользователя: {topic_next}")
                self._users_topics[user_id] = topic_next

            # Добавление ответа
            responses = []
            responses.append(topic_response)

            # Дополнительный ответ?
            if ad:
                responses.append(random.choice(self.bot_config["ad_reponses"]))

            return responses
        # Неизвестная ошибка (на всякий случай)
        except Exception as e:
            logging.error(
                f'Ошибка при генерации сообщения для запроса "{request}" от пользователя {user_id}', exc_info=e
            )
            return [random.choice(self.bot_config["failure"])]
