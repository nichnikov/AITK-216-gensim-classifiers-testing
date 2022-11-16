import queue
import logging
import pathlib
from uuid import uuid4
from utils.data_types import (Etalon,
                              Answer,
                              IdVector)
from pathlib import Path
from ast import literal_eval
from collections import namedtuple

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def get_project_root() -> Path:
    """"""
    return Path(__file__).parent.parent


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def fix_path_to_tests(path: str) -> str:
    path_cwd = pathlib.Path.cwd()
    if "tests" not in path_cwd.parts:
        path_cwd = path_cwd.joinpath("tests")
    else:
        path_cwd = pathlib.Path(*path_cwd.parts[: (path_cwd.parts.index("tests") + 1)])
    return str(path_cwd.joinpath(path))


def queries2etalon(queries: [namedtuple], etalonId_gen=True):
    """"""
    if etalonId_gen:
        return [Etalon(
            templateId=item.templateId,
            etalonText=item.etalonText,
            etalonId=str(uuid4()),
            SysID=item.SysID,
            moduleId=item.moduleId,
            pubsList=item.pubsList
        ) for item in queries]
    else:
        return [Etalon(
            templateId=item.templateId,
            etalonText=item.etalonText,
            etalonId=item.etalonId,
            SysID=item.SysID,
            moduleId=item.moduleId,
            pubsList=item.pubsList
        ) for item in queries]


def expanding_unique(data) -> ():
    """Returns a tuple of unique values for
    [Etalon], [TemplateId], [Answer], [Stopword]"""
    return tuple(set(*zip(*data)))


def answers_fill(data: [namedtuple]) -> [Answer]:
    """"""
    return [Answer(templateId=item.templateId, templateText=item.templateText) for item in data]


def vectors_ids_fill(vectors) -> [IdVector]:
    """"""
    return [IdVector(id=str(uuid4()), vector=vector) for vector in vectors]


def worker_fill(classifier, data, stopwords):
    worker = classifier
    worker.embedding.add_stopwords(stopwords)
    etalons = [Etalon(
        templateId=item.templateId,
        etalonText=item.etalonText,
        etalonId=item.etalonId,
        SysID=item.SysID,
        moduleId=item.moduleId,
        pubsList=item.pubsList
    ) for item in data]

    q = queue.Queue()
    for chunk in chunks(etalons, 2000):
        q.put(chunk)

    while not q.empty():
        worker.add_vectors(q.get())
    return worker


def resulting_report(pubid: int, found_answers_ids: [], text_storage):
    """"""
    if found_answers_ids:
        search_results = text_storage.search_return_all_col(ids=found_answers_ids,
                                                            returned_column_name="pubs",
                                                            table_name="etalons",
                                                            column_name="templateId")

        logger.info("found etalons {}".format(str(search_results)))
        for id_, e_txs, et_ids, s_ids, m_ids, pbs in search_results:
            if pubid in literal_eval(pbs):
                found_answer_texts = text_storage.search_return_one_col(ids=[id_],
                                                                        returned_column_name="templateText",
                                                                        table_name="answers",
                                                                        column_name="templateId")
                found_answer_pubs = text_storage.search_return_one_col(ids=[id_],
                                                                       returned_column_name="pubId",
                                                                       table_name="answers",
                                                                       column_name="templateId")
                logger.info("found_answer_text " + str(found_answer_texts))
                logger.info("found_answer_pubs " + str(found_answer_pubs))
                true_answer = [tx[0] for pb, tx in zip(found_answer_pubs, found_answer_texts) if pb[0] == pubid]
                return {"templateId": id_, "templateText": true_answer[0]}
        return {"templateId": 0, "templateText": ""}
    else:
        logger.info("Empty found_answers_ids")
        return {"templateId": 0, "templateText": ""}
