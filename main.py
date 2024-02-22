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
import json
import logging
import os
import random
import sys
from typing import Dict, List

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
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Telegram bot token from https://t.me/BotFather
BOT_API_KEY = ""

# Start command
BOT_COMMAND_START = "start"

# JSON file with all messages (instead of BOT_CONFIG)
MESSAGES_FILE = "messages.json"

# File with dialogues
DIALOGUES_FILE = "dialogues.txt"

# Limits on how to parse dialogues
DIALOGUES_MAX = 2000
DIALOGUES_MAX_PER_INTENT = 100

# Where to save parsed DIALOGUES_FILE to prevent parsing it each time
DIALOGUES_FILE_PARSED_OUTPUT = "dialogues.json"


# Allowed characters
ALPHABET = "abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890- "


def logging_setup() -> None:
    """Sets up console logging format and level"""

    # Create logs formatter
    log_formatter = logging.Formatter(
        "%(asctime)s %(threadName)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup logging into console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # Add all handlers and setup level
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    # Log test message
    logging.info("Logging setup is complete")


async def _send_safe(
    chat_id: int,
    text: str,
    context: ContextTypes.DEFAULT_TYPE,
    reply_to_message_id=None,
    reply_markup=None,
):
    """Sends message without raising any error

    Args:
        chat_id (int): _description_
        text (str): _description_
        context (ContextTypes.DEFAULT_TYPE): _description_
        reply_to_message_id (_type_, optional): _description_. Defaults to None.
        reply_markup (_type_, optional): _description_. Defaults to None.
    """
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_to_message_id=reply_to_message_id,
            reply_markup=reply_markup,
            disable_web_page_preview=True,
        )
    except Exception as e:
        logging.error(
            f"Error sending {text} to {chat_id}!",
            exc_info=e,
        )


class Dialogue:
    def __init__(self, messages: Dict) -> None:
        self._messages = messages

        self._dialogues_structured = {}

        self._vectorizer = None
        self._classifier = None
        self._users_topics = {}

    def phrase_simplify(self, phrase: str) -> str:
        """Lowers phrase and leaves only ALPHABET characters

        Args:
            phrase (str): request to simplify

        Returns:
            str: simplified request
        """
        return "".join(symbol for symbol in phrase.lower() if symbol in ALPHABET).strip()

    def parse_dialogues_from_file(self) -> None:
        """Loads dialogues from pre-parsed file if exists or parses it
        Parsed dialogues structure:
        (word_1 and word_2 ... are some sort of intents)
        {
            "word_1": [
                [
                    "simplified example",
                    "Answer (response)"
                ],
                [
                    "another simplified example",
                    "Another answer (response)"
                ]
            ],
            "word_2": [
            ....
        }
        """
        # Load from pre-parsed
        if os.path.exists(DIALOGUES_FILE_PARSED_OUTPUT):
            logging.info(f"Loading pre-parsed dialogues from {DIALOGUES_FILE_PARSED_OUTPUT}")
            with open(DIALOGUES_FILE_PARSED_OUTPUT, "r", encoding="utf-8") as file:
                self._dialogues_structured = json.load(file)
            logging.info(f"Dialogues for {len(self._dialogues_structured)} words loaded!")
            return

        # Load the dialogues data from the file
        logging.warning(f"No {DIALOGUES_FILE_PARSED_OUTPUT} file found! Loading dialogues from {DIALOGUES_FILE}")
        with open(DIALOGUES_FILE, "r", encoding="utf-8") as file:
            content = file.read()

        # Split by double lines
        dialogues = [dialogue.split("\n")[:2] for dialogue in content.split("\n\n") if len(dialogue.split("\n")) == 2]

        # Filter out duplicate questions and format the phrases
        logging.info("Filtering dialogues")
        dialogues_filtered = []
        questions = set()
        for dialogue in dialogues:
            # Check max size
            if len(dialogues_filtered) > DIALOGUES_MAX:
                break
            question, answer = dialogue
            question = self.phrase_simplify(question[2:])
            answer = answer[2:]
            if question and question not in questions:
                questions.add(question)
                dialogues_filtered.append([question, answer])

        # Create a dictionary of words and their associated pairs of question and answer
        logging.info("Structurizing dialogues")
        dialogues_structured = {}
        for question, answer in dialogues_filtered:
            words = set(question.split())
            for word in words:
                dialogues_structured.setdefault(word, []).append([question, answer])

        # Sort the pairs by question length and keep only the first DIALOGUES_MAX_PER_INTENT for each word
        logging.info("Sorting dialogues")
        self._dialogues_structured = {
            word: sorted(pairs, key=lambda pair: len(pair[0]))[:DIALOGUES_MAX_PER_INTENT]
            for word, pairs in dialogues_structured.items()
        }

        # Save parsed
        logging.info(f"Saving parsed dialogues to {DIALOGUES_FILE_PARSED_OUTPUT}")
        with open(DIALOGUES_FILE_PARSED_OUTPUT, "w+", encoding="utf-8") as file:
            file.write(json.dumps(self._dialogues_structured, indent=4, ensure_ascii=False))

        # Done
        logging.info(f"Dialogues for {len(self._dialogues_structured)} words loaded!")

    def train_classifier(self, train_on_dialogues: bool) -> None:
        """Creates and trains LinearSVC intent classifier

        Args:
            train_on_dialogues (bool): True to also train using self._dialogues_structured
        """
        intent_names = []
        intent_examples = []

        # Train on dialogues.txt
        if train_on_dialogues:
            for intent, dialogues_list in self._dialogues_structured.items():
                for dialogue_ in dialogues_list:
                    intent_names.append(intent)
                    intent_examples.append(dialogue_[0])

        # Train on normal messages
        for intent, intent_data in self._messages["intents"].items():
            for example in intent_data["examples"]:
                intent_names.append(intent)
                intent_examples.append(self.phrase_simplify(example))

        # Initialize vectorizer
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))

        # Initialize classifier
        if self._classifier is None:
            self._classifier = LinearSVC(dual=True)

        # Train
        logging.info("Training intent classifier...")
        self._classifier.fit(self._vectorizer.fit_transform(intent_examples), intent_names)
        logging.info("Training done!")

    def intent_predict(self, request: str) -> str or None:
        """Tries to predict intent name based on request

        Args:
            request (str): request to predict

        Returns:
            str or None: name of intent or None in case of error or not found
        """
        # Return if not trained
        if self._vectorizer is None or self._classifier is None:
            return

        # Simplify request
        request = self.phrase_simplify(request)
        logging.info(f"Simplified request: {request}")

        # Predict first intent
        intent = self._classifier.predict(self._vectorizer.transform([request]))[0]
        logging.info(f'Predicted intent for "{request}": {intent}')
        return intent

    def next_message(self, request: str, user_id: int) -> List[str]:
        """Generates list of next messages from request and user

        Args:
            request (str): user's new request
            user_id (int): user's id (for storing theme)

        Returns:
            List[str]: messages as strings
        """
        try:
            # List of strings of responses
            responses = []

            # New user
            if not user_id in self._users_topics:
                logging.info(f"Creating new user: {user_id}")
                self._users_topics[user_id] = "undefined"

            # Extract topic
            user_topic = self._users_topics[user_id]

            # Predict intent
            user_intent = self.intent_predict(request)

            # Log request
            logging.info(f"Request from user {user_id}: {request}. User's topic: {user_topic}, intent: {user_intent}")

            # Error
            if not user_intent:
                return [random.choice(self._messages["failure"])]

            # Check in normal messages
            if not user_intent in self._messages["intents"]:
                # Use dialogues.txt instead
                if user_intent in self._dialogues_structured:
                    return [random.choice(self._dialogues_structured[user_intent])[1]]

                # Not in dialogues.txt
                else:
                    return [random.choice(self._messages["failure"])]

            # Check if need to send advertisement
            ad = self._messages["intents"][user_intent]["ad_rate"] > random.random()
            if ad:
                logging.info("Advertising is enabled")

            # Retrieve topic index
            topic_index = 0
            for i, topic in enumerate(self._messages["intents"][user_intent]["topics"]):
                if user_topic in topic["current"]:
                    topic_index = i

            # Select topic
            topic = self._messages["intents"][user_intent]["topics"][topic_index]

            # Select response
            topic_responses = topic["ad"] if (ad and "ad" in topic) else topic["normal"]
            topic_response = random.choice(topic_responses)

            # Retrieve next topic
            topic_next = topic_response["topic_next"]

            # Set user's next topic
            logging.info(f"New user topic: {topic_next}")
            self._users_topics[user_id] = topic_next

            # Fail
            if topic_next == "failure":
                return [random.choice(self._messages["failure"])]

            # Append response
            if "response" in topic_response:
                responses.append(topic_response["response"])

            # Additional response?
            if "additional_intent" in topic_response:
                additional_topics = self._messages["intents"][topic_response["additional_intent"]]["topics"]
                logging.info(f"Additional content: \"{topic_response['additional_intent']}\" intent")
                additional_topic_index = 0
                for i, topic in enumerate(additional_topics):
                    if topic_next in topic["current"]:
                        additional_topic_index = i
                topic_normal = random.choice(additional_topics[additional_topic_index]["normal"])
                if "response" in topic_normal:
                    responses.append(topic_normal["response"])

            return responses

        # Unknown error (just in case)
        except Exception as e:
            logging.error(f'Error generating next message for request "{request}" from user {user_id}', exc_info=e)
            return [random.choice(self._messages["failure"])]


class BotHandler:
    def __init__(self, dialogue: Dialogue) -> None:
        self._dialogue = dialogue

        self._event_loop = None
        self._application = None

    def start(self) -> None:
        """Builds bot and starts polling"""
        # Close previous event loop (just in case)
        try:
            loop = asyncio.get_running_loop()
            if loop and loop.is_running():
                logging.info("Stopping current event loop before starting a new one")
                loop.stop()
        except Exception as e:
            logging.warning(f"Error stopping current event loop: {str(e)}")

        # Create new event loop
        logging.info("Creating new event loop")
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        # Build bot
        builder = ApplicationBuilder().token(BOT_API_KEY)
        self._application = builder.build()

        # Read /start command
        self._application.add_handler(CommandHandler(BOT_COMMAND_START, self._bot_callback_start))

        # Read messages
        self._application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self._bot_callback_message))

        # Start telegram bot polling
        logging.info("Starting bot polling. Press CTRL+C to stop")
        self._application.run_polling(close_loop=True)

    async def _bot_callback_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/start command callback"""
        # Get user's ID
        chat_id = update.effective_chat.id

        # Send start message
        responses = self._dialogue.next_message("start", chat_id)
        for response in responses:
            await _send_safe(chat_id, response, context)

    async def _bot_callback_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Messages callback"""
        # Get user's ID
        chat_id = update.effective_chat.id

        # Extract text request
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

        # Send next message
        responses = self._dialogue.next_message(request_message, chat_id)
        for response in responses:
            await _send_safe(chat_id, response, context)


def main() -> None:
    """Main entry"""
    # Initialize logging
    logging_setup()

    # Load messages from JSON
    logging.info(f"Loading messages from {MESSAGES_FILE}")
    with open(MESSAGES_FILE, "r", encoding="utf-8") as file:
        messages = json.load(file)

    # Initialize dialog handler
    dialogue = Dialogue(messages)

    # Initialize bot class
    bot_handler = BotHandler(dialogue)

    # Parse dialogues
    dialogue.parse_dialogues_from_file()

    # Train
    dialogue.train_classifier(train_on_dialogues=True)

    # Start bot
    bot_handler.start()


if __name__ == "__main__":
    main()
