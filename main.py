import logging
import random

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

BOT_CONFIG = {
    "intents": {
        "hello": {
            "examples": ["Привет", "Добрый день", "Шалом", "Привет, бот"],
            "responses": ["Привет, человек!", "И вам здравствуйте :)", "Доброго времени суток"],
        },
        "bye": {
            "examples": ["Пока", "До свидания", "До свидания", "До скорой встречи"],
            "responses": ["Еще увидимся", "Если что, я всегда тут"],
        },
        "name": {"examples": ["Как тебя зовут?", "Скажи свое имя", "Представься"], "responses": ["Меня зовут Саша"]},
        "want_eat": {
            "examples": ["Хочу есть", "Хочу кушать", "ням-ням"],
            "responses": ["Вы веган?"],
            "theme_gen": "eating_q_wegan",
            "theme_app": ["eating", "*"],
        },
        "yes": {
            "examples": ["да"],
            "responses": ["капусты или морковки?"],
            "theme_gen": "eating_q_meal",
            "theme_app": ["eating_q_wegan"],
        },
        "no": {
            "examples": ["нет"],
            "responses": ["мясо или творог?"],
            "theme_gen": "eating_q_meal",
            "theme_app": ["eating_q_wegan"],
        },
    },
    "failure_phrases": [
        "Непонятно. Перефразируйте, пожалуйста.",
        "Я еще только учусь. Спросите что-нибудь другое",
        "Слишком сложный вопрос для меня.",
    ],
}

X_text = []  # ['Хэй', 'хаюхай', 'Хаюшки', ...]
y = []  # ['hello', 'hello', 'hello', ...]

for intent, intent_data in BOT_CONFIG["intents"].items():
    for example in intent_data["examples"]:
        X_text.append(example)
        y.append(intent)

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC(dual=True)
clf.fit(X, y)


def clear_phrase(phrase):
    phrase = phrase.lower()

    alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя- "
    result = "".join(symbol for symbol in phrase if symbol in alphabet)

    return result.strip()


def classify_intent(replica):
    replica = clear_phrase(replica)

    intent = clf.predict(vectorizer.transform([replica]))[0]

    for example in BOT_CONFIG["intents"][intent]["examples"]:
        example = clear_phrase(example)
        distance = nltk.edit_distance(replica, example)
        if example and distance / len(example) <= 0.5:
            return intent


def get_answer_by_intent(intent):
    if intent in BOT_CONFIG["intents"]:
        responses = BOT_CONFIG["intents"][intent]["responses"]
        if responses:
            return random.choice(responses)


with open("dialogues.txt") as f:
    content = f.read()

dialogues_str = content.split("\n\n")
dialogues = [dialogue_str.split("\n")[:2] for dialogue_str in dialogues_str]

dialogues_filtered = []
questions = set()

for dialogue in dialogues:
    if len(dialogue) != 2:
        continue

    question, answer = dialogue
    question = clear_phrase(question[2:])
    answer = answer[2:]

    if question != "" and question not in questions:
        questions.add(question)
        dialogues_filtered.append([question, answer])

dialogues_structured = {}  #  {'word': [['...word...', 'answer'], ...], ...}

for question, answer in dialogues_filtered:
    words = set(question.split(" "))
    for word in words:
        if word not in dialogues_structured:
            dialogues_structured[word] = []
        dialogues_structured[word].append([question, answer])

dialogues_structured_cut = {}
for word, pairs in dialogues_structured.items():
    pairs.sort(key=lambda pair: len(pair[0]))
    dialogues_structured_cut[word] = pairs[:1000]

# replica -> word1, word2, word3, ... -> dialogues_structured[word1] + dialogues_structured[word2] + ... -> mini_dataset


def generate_answer(replica):
    replica = clear_phrase(replica)
    words = set(replica.split(" "))
    mini_dataset = []
    for word in words:
        if word in dialogues_structured_cut:
            mini_dataset += dialogues_structured_cut[word]

    # TODO убрать повторы из mini_dataset

    answers = []  # [[distance_weighted, question, answer]]

    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < 0.2:
                answers.append([distance_weighted, question, answer])

    if answers:
        return min(answers, key=lambda three: three[0])[2]


def get_failure_phrase():
    failure_phrases = BOT_CONFIG["failure_phrases"]
    return random.choice(failure_phrases)


stats = {"intent": 0, "generate": 0, "failure": 0}


def bot(replica):
    # NLU
    intent = classify_intent(replica)

    # Answer generation

    # выбор заготовленной реплики
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats["intent"] += 1
            return answer

    # вызов генеративной модели
    answer = generate_answer(replica)
    if answer:
        stats["generate"] += 1
        return answer

    # берем заглушку
    stats["failure"] += 1
    return get_failure_phrase()


print(bot("Сколько времени?"))

############### ТЕЛЕГРАММ ###########################

# https://github.com/python-telegram-bot/python-telegram-bot


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi!")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Help!")


async def run_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    replica = update.message.text
    answer = bot(replica)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

    print(stats)
    print(replica)
    print(answer)
    print()


def main():
    """Start the bot."""
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    builder = ApplicationBuilder().token("6909462391:AAHpU3Qw8U-jugE7p5VoGvkxeo0ltmFvVeY")
    application = builder.build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), run_bot))

    # Start the Bot
    application.run_polling(close_loop=True)


main()
