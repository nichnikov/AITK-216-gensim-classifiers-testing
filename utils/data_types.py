from typing import NamedTuple, List, Literal
from scipy.sparse import csc_matrix
from pydantic import BaseModel, Field
from collections import namedtuple

Query = namedtuple("Query", "templateId, etalonText, etalonId, SysID, moduleId, pubsList")
FastAnswer = namedtuple("FastAnswer", "templateId, templateText, pubId")
ROW_FOR_ANSWERS = namedtuple("ROW", "SysID, ID, ParentModuleID, ParentID, ChildBlockModuleID, ChildBlockID")

ROW = namedtuple("ROW", "SysID, ID, Cluster, ParentModuleID, ParentID, ParentPubList, "
                          "ChildBlockModuleID, ChildBlockID, ModuleID")


class IdVector(NamedTuple):
    """"""
    id: str
    vector: csc_matrix


# class Etalon(NamedTuple):
class Etalon(BaseModel):
    """"""
    templateId: int
    etalonText: str
    etalonId: str
    SysID: int
    moduleId: int
    pubsList: str


class TemplateId(BaseModel):
    templateId: int


class TemplateIds(BaseModel):
    templateIds: List[int]
    # pubId: int


class Answer(BaseModel):
    templateId: int
    templateText: str
    pubId: int


# class Stopword(BaseModel):
#    stopword: str


class Etalons(BaseModel):
    etalons: List[Etalon]
    SysID: int


class Answers(BaseModel):
    """"""
    answers: List[Answer]


class Stopwords(BaseModel):
    """"""
    stopwords: List[str]


class SearchData(BaseModel):
    """"""
    pubid: int = Field(title="Пабайди, в котором будет поиск дублей")
    text: str = Field(title="вопрос для поиска")
