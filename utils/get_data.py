"""
Данные берутся из таблицы БД Statistics и формируется правильная ссылка
"""
import os
import requests
import logging
import pandas as pd
from uuid import uuid4
from src.data_types import (Etalon,
                            Answer,
                            Query,
                            ROW_FOR_ANSWERS,
                            ROW,
                            FastAnswer)



def get_data_from_csv(DATA_PATH, pubs: [], sysId: int):
    """Function, getting data from csv file with fields:

        templateId
        text
        templateText

        and returned list of Etalons and list of Answers
     """
    appendix = 100000000 * int(sysId)
    df = pd.read_csv(DATA_PATH, sep="\t")
    etln_dcts = df.to_dict(orient="records")
    etalons_from_csv = [Etalon(templateId=row["templateId"] + appendix,
                               etalonText=row["text"],
                               etalonId=str(uuid4()),
                               SysID=sysId,
                               moduleId=99,
                               pubsList=str(pubs)) for row in etln_dcts]

    answers_tuples = []
    for pub in pubs:
        answers_tuples += [FastAnswer(row["templateId"] + appendix,
                                      row["templateText"], pub) for row in etln_dcts]
    answers_from_csv = [Answer(templateId=x.templateId,
                               templateText=x.templateText,
                               pubId=x.pubId) for x in set(answers_tuples)]

    return etalons_from_csv, answers_from_csv


def upload_data_from_csv(**kwargs):
    """Загрузка данных из CSV файла:"""
    etalons_list, answers_list = get_data_from_csv(kwargs["DATA_PATH"], kwargs["pubs"], kwargs["SysID"])

    """deleting etalons:"""
    etalons_templates = [e.templateId for e in etalons_list]
    response = requests.post(kwargs["etalons_delete_url"], json={"templateIds": etalons_templates})

    """adding etalons:"""
    etalons = [e.dict() for e in etalons_list]
    et_add_data = {"SysID": kwargs["SysID"], "etalons": etalons}
    response = requests.post(kwargs["etalons_add_url"], json=et_add_data)

    """deleting answers:"""
    answers_templates = [a.templateId for a in answers_list]
    response = requests.post(kwargs["answer_delete_url"], json={"templateIds": answers_templates})

    """adding answers:"""
    answers = [a.dict() for a in answers_list]
    asw_add_data = {"answers": answers}
    response = requests.post(kwargs["answer_add_url"], json=asw_add_data)
    return 0
