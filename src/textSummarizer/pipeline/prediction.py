from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.conponents.Translator import TranslatorService
from src.textSummarizer.logging import logger
from src.textSummarizer.exceptions import *
from src.textSummarizer.utils.common import get_mysql_db
# from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from transformers import pipeline
from fastapi import HTTPException

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.output = ""


    def predict(self,text):
        try:
            logger.info("Original Text:")
            logger.info(text)
            db = get_mysql_db()
            translator_service = TranslatorService()
            translated_text = translator_service.get_translation(db, text)

            if translated_text is None:
                translation_text = translator_service.perform_translation(db, text)

                # # Save the translation to database
                # translator_service.save_translation(text, source_lang, target_lang, translation_text)

                # logger.info("Translation not available in DB, generating translation...")
                # model = M2M100ForConditionalGeneration.from_pretrained(self.config.tokenizer_path)
                # tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

                # text_to_translate = text
                # model_inputs = tokenizer(text_to_translate, return_tensors="pt")

                # # translate to French
                # gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("fr"))
                # output = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                # output = output[0]
                # logger.info("\nTranslated Text:")
                # logger.info(output)

                # Save the translation to database
                translator_service.save_translation(db, text, translation_text)

                self.output = translation_text
            else:
                self.output = translated_text
            
            return self.output

        except TranslationServiceException as tse:
            logger.error(f"Translation service error: {tse}")
            raise HTTPException(status_code=500, detail=str(tse))
        except DatabaseException as de:
            logger.error(f"Database error: {de}")
            raise HTTPException(status_code=500, detail=str(de))
