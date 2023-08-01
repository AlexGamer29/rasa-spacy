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


@lru_cache()
def load_model(model_file) -> None:
    logging.warning(f"========= load_model:  ========")
    if os.path.exists(model_file):
        return ahocorasick.load(model_file, pickle.loads)
    return ahocorasick.Automaton()


class SpellChecker:
    def __init__(self) -> None:
        self.model_file = f"./data_ext/spell_ar_07.pickle"
        self.model = load_model(self.model_file)
        self.ar_alphabet = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىـ"
        self.distance = 2

    def generate_edit_distance(self, word: str):
        # n =3
        logging.warning(f"========= generate_edit_distance:  ========")
        cache = {0: set([word])}
        for i in range(1, self.distance + 1):
            cache[i] = set()
            for w in cache[i - 1]:
                splits = [(w[:j], w[j:]) for j in range(len(w) + 1)]
                cache[i] |= set(
                    [L + R[1:] for L, R in splits if R]
                    + [L + R[0] + R[2:] for L, R in splits if len(R) > 1]
                    + [L + c + R[1:] for L, R in splits if R for c in self.ar_alphabet]
                    + [L + c + R for L, R in splits for c in self.ar_alphabet]
                )
        return cache

    def spell_check(self, txt: str, mode: int) -> str:
        logging.warning(f"========= generate_edit_distance:  ========")
        A = self.model
        words = txt.split()
        text_new = ""
        for w in words:
            if A.exists(w) and mode == 0:
                text_new = f"{text_new} {w}"
                continue
            selected_rank = 0
            selected_correction = ""
            edits_dict = self.generate_edit_distance(w)
            for key, value in edits_dict.items():
                for edit in value:
                    if A.exists(edit):
                        val = A.get(edit, {})
                        rank = val.get("r", 0)
                        if rank > selected_rank:
                            selected_correction = edit
                            selected_rank = rank
            if selected_rank == 0:
                selected_correction = w
            text_new = f"{text_new} {selected_correction}"
        return text_new
