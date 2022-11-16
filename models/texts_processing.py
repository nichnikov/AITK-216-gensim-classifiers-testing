import re
# from hunspell import Hunspell
from pymystem3 import Mystem
from itertools import chain, filterfalse


class TextsTokenizer:
    """Tokenizer"""

    def __init__(self):
        self.stopwords = []
        self.m = Mystem()

    def texts2tokens(self, texts: [str]) -> [str]:
        """Lemmatization for texts in list. It returns list with lemmatized texts"""
        text_ = "\n".join(texts)
        text_ = re.sub(r"[^\w\n\s]", " ", text_)
        lm_texts = "".join(self.m.lemmatize(text_))
        return [lm_tx.split() for lm_tx in lm_texts.split("\n")][:-1]

    def add_stopwords(self, stopwords: [str]):
        """adding stop words into class"""
        self.stopwords += [x for x in chain(*self.texts2tokens(stopwords))]

    def del_stopwords(self, stopwords: [str]):
        """adding stop words into class"""
        stopwords_del = [x for x in chain(*self.texts2tokens(stopwords))]
        self.stopwords = [w for w in self.stopwords if w not in stopwords_del]


    def tokenization(self, texts: [str]) -> [[]]:
        """list of texts lemmatization with stop words deleting"""
        lemm_texts = self.texts2tokens(texts)
        return [list(filterfalse(lambda x: x in self.stopwords, tx)) for tx in lemm_texts]

    def __call__(self, texts: [str]):
        return self.tokenization(texts)

'''
class SpellChecker:
    def __init__(self, dict_path):
        self.h = Hunspell("ru_RU", hunspell_data_dir=dict_path)

    def spell_checking(self, text: str) -> str:
        """"""
        checking_dict = self.h.bulk_suggest(text.split())
        return " ".join([checking_dict[w][0] for w in checking_dict])

    def __call__(self, text):
        return self.spell_checking(text)
'''