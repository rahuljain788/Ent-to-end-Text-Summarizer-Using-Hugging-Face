import re
import logging
import time
import torch
import nltk
import spacy
import en_core_web_md
import fr_core_news_md
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import MarianTokenizer, MarianMTModel
from src.textSummarizer.exceptions import TranslationServiceException, DatabaseException
from src.textSummarizer.logging import logger


class TranslatorService:
    def __init__(self):
        nltk.download('punkt')
        # self.nlp_fr = spacy.load('fr_core_news_md')
        # self.nlp_eng = spacy.load('en_core_web_md')
        logger.info("Loading models...")
        # self.nlp_fr = spacy.load("fr_core_news_md")
        # self.nlp_eng = spacy.load("en_core_web_md")
        self.nlp_fr = fr_core_news_md.load()
        self.nlp_eng = en_core_web_md.load()
        # self.model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418m')
        # self.tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418m')
        model_name = "Helsinki-NLP/opus-mt-en-fr"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        logger.info("Models loaded successfully.")

    def get_translation(self, db, text, source_lang='en', target_lang='fr'):
        """
        Get available translation from the database or return None.

        :param db: MySQLDatabase instance.
        :param text: Text to be translated.
        :param source_lang: Source language code.
        :param target_lang: Target language code.

        :return: Translation result or None if not found.
        """
        try:
            logger.info("Fetching existing translation from the database")

            query = f"""SELECT
                            fs.french_sentence AS translated_text
                        FROM
                            english_sentences AS es
                        JOIN french_sentences AS fs ON
                            es.french_sentence_id = fs.id
                        WHERE
                            es.english_sentence = '{text}';"""

            result = db.execute_query(query)
            logger.info(f"Result of {query}: is {result}")

            if result:
                logger.info(f"Translation retrieved from the database: {result}")
                return result[0]['translated_text']
            else:
                logger.info("Translation not found in the database")
                return None

        except Exception as e:
            logging.error(f"Error in get_translation: {e}")
            raise DatabaseException("Database error")

    def perform_translation(self, db, text, source_lang='en', target_lang='fr'):
        """
        Perform text translation using NLP.
        :param text: Text to be translated.
        :param source_lang: Source language code.
        :param target_lang: Target language code.
        :return: Translation result.
        """
        try:
            logger.info(f"Performing translation from {source_lang} to {target_lang} on the text: {text}")
            # Performing translation...
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            logger.info("Check if translation need to be done on a word or sentence...")
            if len(sentences) == 1 and len(words) == 1 and not words[0].isnumeric() and not words[0].isalpha():
                logger.info("Performing Word translation...")
                # tokens = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
                # output = self.model.generate(**tokens, forced_bos_token_id=self.tokenizer.get_lang_id(target_lang))
                # translated_word = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                # translated_word = ''.join(translated_word)

                # Tokenize the input text and translate it
                inputs = self.tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    output = self.model.generate(**inputs)
                    # Decode the translated output
                translated_word = self.tokenizer.decode(output[0], skip_special_tokens=True)

                return translated_word
            else:
                logger.info("Performing Sentence translation...")
                logger.info("Fetch the Similar sentence from DB...")
                # TODO - Implement functionality to check If only nouns are differentiated between 2 sentences
                closest_translation = self.fetch_most_similar_translation_from_db(db, text)

                if closest_translation:
                    # Noun Replacement
                    logger.info("Similar translation found. Performing noun replacement...")
                    # noun_replaced_text = self.replace_nouns(db, text, closest_translated_text)
                    noun_replaced_text = self.replace_nouns_updated(db, text, closest_translation)
                    if noun_replaced_text:
                        translated_text = noun_replaced_text
                        logger.info(f"Translation performed successfully and the translated text is: {translated_text}")
                        return translated_text

                # Sentence translation using NLP model
                logger.info("Similar translation not found. So Performing sentence translation using NLP model...")
                # tokens = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
                # output = self.model.generate(**tokens, forced_bos_token_id=self.tokenizer.get_lang_id(target_lang))
                # translated_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)

                # Tokenize the input text and translate it
                inputs = self.tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    output = self.model.generate(**inputs)
                    # Decode the translated output
                translated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

                logger.info(f"Translation performed successfully and the translated text is: {translated_text}")
                self.save_noun_replacements(db, text, translated_text)
                return translated_text

        except Exception as e:
            print(f"Error in perform_translation: {e}")
            raise TranslationServiceException("Translation service error")

    def fetch_most_similar_translation_from_db(self, db, input_sentence):
        # Variables to keep track of the most similar sentence and its similarity score
        most_similar_fr_translation_id = None
        most_similar_en_sentence = None
        highest_similarity_score = -1.0

        # Retrieve existing French sentences from the table
        existing_translations = db.execute_query(
            "SELECT french_sentence_id, english_sentence FROM english_sentences")

        # Checking similarity with each existing sentence
        for existing_translation in existing_translations:
            similarity_score = self.nlp_eng(input_sentence).similarity(
                self.nlp_eng(existing_translation['english_sentence']))

            # Check if similarity is above the threshold and higher than the current highest score
            if similarity_score >= 0.90 and similarity_score > highest_similarity_score:
                highest_similarity_score = similarity_score
                most_similar_fr_translation_id = existing_translation['french_sentence_id']
                most_similar_en_sentence = existing_translation['english_sentence']

        result = db.execute_query(
            f"SELECT french_sentence FROM french_sentences where id='{most_similar_fr_translation_id}'")
        most_similar_translated_text = result[0]['french_sentence'] if result else None

        if most_similar_translated_text and most_similar_en_sentence:
            similar_translation = {'english_sentence': most_similar_en_sentence,
                                   'french_sentence': most_similar_translated_text}
            return similar_translation

    def extract_en_nouns(self, en_sentence):
        """
        Function to extract nouns from the input English sentence.
        :param en_sentence: input English sentence
        :return: English nouns
        """
        en_nouns = [token.text for token in self.nlp_eng(str(en_sentence)) if token.pos_ in ['NOUN', 'PROPN']]
        return en_nouns

    def extract_fr_nouns(self, fr_sentence):
        """
        Function to extract nouns from the input French sentence.
        :param fr_sentence: input French sentence
        :return: French nouns
        """
        fr_nouns = [token.text for token in self.nlp_fr(str(fr_sentence)) if token.pos_ in ['NOUN', 'PROPN']]
        return fr_nouns

    def lookup_fr_noun(self, db, eng_noun):
        """
        Function to lookup existing French nouns based on English nouns
        :param eng_noun:
        :return: french noun or None
        """
        # Implement a method to lookup French nouns based on English nouns
        try:
            logger.info("Looking for existing translation of a English noun in database...")

            query = f"""SELECT french_noun FROM noun_replacements WHERE english_noun = '{eng_noun}';"""

            result = db.execute_query(query)

            if result:
                logger.info(f"Noun retrieved from the database: {result}")
                return result[0]['french_noun']
            else:
                logger.info("Noun not found in the database")
                return None

        except Exception as e:
            logger.error(f"Error in lookup_fr_noun: {e}")
            raise DatabaseException("Database error")

    def replace_nouns(self, db, input_english_sentence, closest_french_translation):
        """
        Function to replace nouns in the closest French translation of a given English sentence
        :param input_english_sentence:
        :param closest_french_translation:
        :return:
        """
        input_english_sentence_nouns = self.extract_en_nouns(input_english_sentence)
        closest_french_sentence_nouns = self.extract_fr_nouns(closest_french_translation)

        if len(input_english_sentence_nouns) == len(closest_french_sentence_nouns):
            for en_noun, fr_noun_to_place in zip(input_english_sentence_nouns, closest_french_sentence_nouns):
                # Check for existing noun replacements
                fr_noun = self.lookup_fr_noun(db, en_noun)
                if fr_noun is not None:
                    # Noun replacement found in database
                    if fr_noun in closest_french_translation:
                        # Skipping if noun already present in translation
                        pass
                    else:
                        # Replace the noun in the translation
                        closest_french_translation = closest_french_translation.replace(
                            fr_noun_to_place, fr_noun)
                else:
                    # Existing french noun replacement not found. So, fetching the noun translation using NLP model
                    fr_noun = self.perform_translation(db, text=en_noun)
                    closest_french_translation = closest_french_translation.replace(
                        fr_noun_to_place, fr_noun)
                    # Adding the translated noun to Noun Replacement table in database
                    self.save_word_translation(db, en_noun, fr_noun)

            # Replace numbers
            # Regular expression pattern to match numbers with optional decimal point or comma
            pattern = r'(\d+(?:[.,]\d+)?)'

            # Find all numbers in each sentence
            numbers1 = re.findall(pattern, input_english_sentence)
            numbers2 = re.findall(pattern, closest_french_translation)

            # Checking if the sentence is having same number of
            if len(numbers1) + len(numbers2) != 0 and len(numbers1) == len(numbers2):
                for num2, num1 in zip(numbers2, numbers1):
                    closest_french_translation = closest_french_translation.replace(num2, num1)

                # Return the translation only when all the nouns and numbers are replaced properly
                return closest_french_translation

    def replace_nouns_updated(self, db, input_english_sentence, closest_translation):
        """
        Function to replace nouns in the closest French translation of a given English sentence
        :param input_english_sentence:
        :param closest_translation:
        :return:
        """
        input_english_sentence_nouns = self.extract_en_nouns(input_english_sentence)
        closest_english_sentence_nouns = self.extract_en_nouns(closest_translation['english_sentence'])
        closest_french_translation = closest_translation['french_sentence']
        logger.info("input_english_sentence_nouns: " + str(input_english_sentence_nouns))
        logger.info("closest_english_sentence_nouns: " + str(closest_english_sentence_nouns))
        logger.info("closest_french_translation: " + str(closest_french_translation))
        # closest_french_sentence_nouns = self.extract_fr_nouns(closest_french_translation)

        if len(input_english_sentence_nouns) == len(closest_english_sentence_nouns):
            for en_noun, closest_en_noun in zip(input_english_sentence_nouns, closest_english_sentence_nouns):
                if en_noun == closest_en_noun:
                    pass
                else:
                    # Check for existing noun replacements
                    fr_noun = self.lookup_fr_noun(db, en_noun)
                    existing_fr_noun = self.lookup_fr_noun(db, closest_en_noun) if len(closest_en_noun) > 1 else closest_en_noun
                    if fr_noun is not None and existing_fr_noun is not None:
                        if existing_fr_noun is not None:
                            # Replace the noun in the translation
                            closest_french_translation = closest_french_translation.replace(
                                existing_fr_noun, fr_noun)
                    else:
                        # Existing french noun replacement not found. So, fetching the noun translation using NLP model
                        fr_noun = self.perform_translation(db, text=en_noun) if len(en_noun) > 1 else en_noun

                        closest_french_translation = closest_french_translation.replace(
                            existing_fr_noun, fr_noun)
                        # Adding the translated noun to Noun Replacement table in database
                        self.save_word_translation(db, en_noun, fr_noun)

            # Replace numbers
            # Regular expression pattern to match numbers with optional decimal point or comma
            pattern = r'(\d+(?:[.,]\d+)?)'

            # Find all numbers in each sentence
            numbers1 = re.findall(pattern, input_english_sentence)
            numbers2 = re.findall(pattern, closest_french_translation)

            # Checking if the sentence is having same number of
            if len(numbers1) + len(numbers2) != 0 and len(numbers1) == len(numbers2):
                for num2, num1 in zip(numbers2, numbers1):
                    closest_french_translation = closest_french_translation.replace(num2, num1)

            # Return the translation only when all the nouns and numbers are replaced properly
            return closest_french_translation

    def save_word_translation(self, db, word, translated_word):
        """
        Save the word translation in database.
        :param word: Word to be translated.
        :param translated_word: Translation result.
        """
        try:
            uid = int(time.time())  # Unique ID for each word replacement
            logger.info(f"Unique ID created for word translation: {uid}")

            # Add Word translation to database
            logger.info("Adding word translation to the database")
            add_word_query = "INSERT IGNORE INTO noun_replacements (id, english_noun, french_noun) VALUES (%s, %s, %s);"
            fr_params = (uid, word, translated_word.replace("'", "\'"))
            db.execute_query(add_word_query, fr_params)
            logger.info("Word translation added to the database")

            # commit the data
            db.connection.commit()
            logger.info("Word translation saved to the database")

        except Exception as e:
            logger.error(f"Error in save_word_translation: {e}")
            raise DatabaseException("Database error")

    def save_noun_replacements(self, db, input_text, translated_text):
        """
        Save all the noun replacements taken from a translation into database.
        :param input_text: Original input text received for translation.
        :param translated_text: Translated text.
        """
        try:
            logger.info(f"Fetching nouns from both input text and translated text...")
            en_nouns = self.extract_en_nouns(en_sentence=input_text)
            fr_nouns = self.extract_fr_nouns(fr_sentence=translated_text)
            logger.info("Creating noun replacements mapping...")
            # Filter already existing nouns replacements

            noun_replacement_params = []
            uid = int(time.time())  # Unique ID
            for en_noun, fr_noun in zip(en_nouns, fr_nouns):
                uid = uid + 1
                noun_replacement = (uid, en_noun, fr_noun)
                noun_replacement_params.append(noun_replacement)
            logger.info(f"Noun replacements mapping: {noun_replacement_params}")

            logger.info("Adding noun replacements to database")
            add_noun_replacements_query = "INSERT IGNORE INTO noun_replacements (id, english_noun, french_noun) VALUES (%s, %s, %s)"
            db.execute_many(add_noun_replacements_query, noun_replacement_params)
            logger.info("Noun replacements added to the database")

            # commit the data
            db.connection.commit()
            logger.info("Noun replacements saved to the database")

        except Exception as e:
            logger.error(f"Error in save_noun_replacements: {e}")
            raise DatabaseException("Database error")

    def save_translation(self, db, text, translation_text):
        """
        Save the translation in database.
        :param db: MySQLDatabase instance.
        :param text: Text to be translated.
        :param source_lang: Source language code.
        :param target_lang: Target language code.
        :param translation_text: Translation result.
        """
        try:
            uid = int(time.time())  # Unique Id for each translation
            logger.info(f"Unique ID created for translation: {uid}")

            # Add French translation to database
            logger.info("Adding French translation to the database")
            add_fr_sentence_query = "INSERT INTO french_sentences (id, french_sentence) VALUES (%s, %s);"
            fr_params = (uid, translation_text)
            db.execute_query(add_fr_sentence_query, fr_params)
            logger.info("French translation added to the database")

            # Add English sentence to database
            logger.info("Adding the original English sentence to database")
            add_en_sentence_query = "INSERT INTO english_sentences (id, english_sentence, french_sentence_id) VALUES (%s, %s, %s);"
            en_params = (uid, text, uid)
            db.execute_query(add_en_sentence_query, en_params)
            logger.info("English sentence added to the database")

            # commit the data
            db.connection.commit()
            logger.info("Translation saved to the database")

        except Exception as e:
            logging.error(f"Error in save_translation: {e}")
            raise DatabaseException("Database error")
