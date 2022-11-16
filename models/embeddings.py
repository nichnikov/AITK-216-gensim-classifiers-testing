from models.texts_processing import TextsTokenizer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
import copy


class TokensVectorsBoW:
    """"""

    def __init__(self, max_dict_size: int):
        self.dictionary = None
        self.max_dict_size = max_dict_size

    def tokens2corpus(self, tokens: []):
        """queries2vectors new_queries tuple: (text, query_id)
        return new vectors with query ids for sending in searcher"""

        if self.dictionary is None:
            # TODO adding len gensim_dict in logging
            gensim_dict_ = Dictionary(tokens)
            assert len(gensim_dict_) <= self.max_dict_size, "len(gensim_dict) must be less then max_dict_size"
            self.dictionary = Dictionary(tokens)
        else:
            gensim_dict_ = copy.deepcopy(self.dictionary)
            gensim_dict_.add_documents(tokens)
            if len(gensim_dict_) <= self.max_dict_size:
                self.dictionary = gensim_dict_
        return [self.dictionary.doc2bow(lm_q) for lm_q in tokens]

    def tokens2vectors(self, tokens: []):
        """"""
        corpus = self.tokens2corpus(tokens)
        return [corpus2csc([x], num_terms=self.max_dict_size) for x in corpus]

    def __call__(self, new_tokens):
        return self.tokens2vectors(new_tokens)


class TokensVectorsTfIdf(TokensVectorsBoW):
    """"""

    def __init__(self, max_dict_size):
        super().__init__(max_dict_size)
        self.tfidf_model = None

    def model_fill(self, tokens: []):
        """"""
        assert self.tfidf_model is None, "the model is already filled"
        corpus = super().tokens2corpus(tokens)
        self.tfidf_model = TfidfModel(corpus)

    def tokens2vectors(self, tokens: []):
        """"""
        vectors = super().tokens2corpus(tokens)
        return [corpus2csc([x], num_terms=self.max_dict_size) for x in self.tfidf_model[vectors]]

    def __call__(self, new_tokens):
        return self.tokens2vectors(new_tokens)


class BowEmbedding:
    """
    Embedding with Bag of Words
    """

    def __init__(self, VOCABULARY_SIZE):
        self.tokenizer = TextsTokenizer()
        self.vectorizer = TokensVectorsBoW(max_dict_size=VOCABULARY_SIZE)

    def add_stopwords(self, stop_words_list: []):
        self.tokenizer.add_stopwords(stop_words_list)

    def del_stopwords(self, stop_words_list: []):
        self.tokenizer.del_stopwords(stop_words_list)

    def vectors_maker(self, texts: [str]) -> []:
        """"""
        tokens = self.tokenizer(texts)
        return self.vectorizer(tokens)

    def __call__(self, texts):
        return self.vectors_maker(texts)


'''
class TransformerEmbedding:
    """"""
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    def vectors_maker(self, texts: [str]) -> []:
        """"""
        return self.model.encode([x.lower() for x in texts])

    def __call__(self, texts):
        return self.vectors_maker(texts)'''
