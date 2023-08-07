from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import logging
from functools import lru_cache
from datetime import datetime
import ahocorasick
import pickle
import os
# from spell_checker import SpellChecker
import spacy
from spacy.tokens import Doc
from spellchecker import SpellChecker
from rasa_sdk import Action, Tracker

# TODO: Correctly register your component with its type
@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER], is_trainable=False
)
class CustomNLUComponent(GraphComponent):
    def __init__(self) -> None:
        super().__init__()  
        # self.load_model()
        # self.ar_alphabet = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىـ"
        # self.model_file = f"./data_ext/spell_ar_07.pickle"
        # self.model = load_model(self.model_file)

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # TODO: Implement this if your component augments the training data with
        #       tokens or message features which are used by other components
        #       during training.
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        nlp = spacy.load("en_core_web_md")
        spell = SpellChecker()
        for message in messages:
            logging.info(
                f"############### before: {message.data['text']} ############### "
            )
            msg_text = message.data["text"]
            # Check for named entities (addressed as "GPE" in SpaCy)
            doc = nlp(msg_text)
            named_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

            # Spell check and correct the text
            # corrected_text = spell_checker.spell_check(msg_text, 0)
            words = msg_text.split()
            # corrected_text = self.auto_correct_spelling(msg_text)
            corrected_words = [spell.correction(word) for word in words]
            corrected_text = ' '.join(corrected_words)


            # proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
            person_names = [entity.text for entity in doc.ents if entity.label_ == "PERSON"]
            # Response if it detect location 
            if named_entities:
                response = f"Here are the extracted locations: {', '.join(named_entities)}"
            else:
                response = "No locations were extracted from the input."

            logging.info(response)
            logging.info(person_names)

            logging.info(f"############### after: {corrected_text} ############### ")

            message.data["text"] = corrected_text

        return messages