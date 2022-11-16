import logging
from itertools import chain
from multiprocessing import Pool

from scipy.sparse import hstack, vstack
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_types import IdVector
from utils.utils import chunks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class IdsMatrix:
    """"""

    def __init__(self):
        self.ids = []
        self.matrix = None

    def delete(self, ids: [str]) -> None:
        ids_vectors = [(i, v) for i, v in zip(self.ids, self.matrix) if i not in ids]
        if ids_vectors:
            self.ids, vectors = zip(*ids_vectors)
            self.matrix = vstack(vectors)
        else:
            self.ids = []
            self.matrix = None

    def add(self, ids_vectors: [IdVector]) -> None:
        """tuples must be like (text_id, text_vector)"""
        ids, vectors = zip(*ids_vectors)
        self.ids += ids
        if self.matrix is None:
            self.matrix = hstack(vectors).T
        else:
            self.matrix = vstack((self.matrix, hstack(vectors).T))


class MatricesList:
    """"""

    def __init__(self, max_size: int):
        self.ids_matrix_list = [IdsMatrix()]
        self.max_size = max_size

    @property
    def quantity(self) -> int:
        _sum = sum([len(m.ids) for m in self.ids_matrix_list])
        logger.info(f"quantity: {_sum}")
        return _sum

    @staticmethod
    def search_func(data: dict) -> list:
        """searched_vectors tuples must be like (query_id, query_vector)"""
        vectors_ids = data["vectors_ids"]
        vectors = data["vectors"]
        matrix = data["matrix"]
        matrix_ids = data["matrix_ids"]
        score = data["score"]

        searched_matrix = hstack(vectors).T
        if matrix is None:
            return []
        try:
            matrix_scores = cosine_similarity(searched_matrix, matrix, dense_output=False)
            search_results = [
                [(v_id, matrix_ids[mrx_i], sc) for mrx_i, sc in zip(scores.indices, scores.data) if sc >= score]
                for v_id, scores in zip(vectors_ids, matrix_scores)
            ]
            results = [x for x in chain(*search_results) if x]
            logger.info("Searching successfully completed with results: {}".format(str(results)))
            return results
        except Exception as e:
            logger.error("Failed queries search in MainSearcher.search: ", str(e))
            return []

    def add(self, ids_vectors: [IdVector]) -> None:
        """"""
        for chunk in chunks(ids_vectors, self.max_size):
            is_matrices_full = True
            for im in self.ids_matrix_list:
                if len(im.ids) < self.max_size:
                    im.add(chunk)
                    is_matrices_full = False
            if is_matrices_full:
                """adding new queries_matrix"""
                im = IdsMatrix()
                im.add(chunk)
                self.ids_matrix_list.append(im)

    def delete(self, ids: [str]) -> None:
        """"""
        for ids_matrix in self.ids_matrix_list:
            if set(ids) & set(ids_matrix.ids):
                ids_matrix.delete(ids)

    def search(self, searched_vectors: [IdVector], min_score: float) -> [tuple]:
        vectors_ids, vectors = zip(*searched_vectors)
        searched_data = [
            {
                "vectors_ids": vectors_ids,
                "vectors": vectors,
                "matrix": mx.matrix,
                "matrix_ids": mx.ids,
                "score": min_score,
            }
            for mx in self.ids_matrix_list
        ]
        # logger.info("searched_data: {}".format(str(searched_data)))
        pool = Pool()
        search_result = pool.map(self.search_func, searched_data)
        pool.close()
        pool.join()
        logger.info("search_result: {}".format(str(search_result)))
        return [x for x in chain(*search_result) if x]
