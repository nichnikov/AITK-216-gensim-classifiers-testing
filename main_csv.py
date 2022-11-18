# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import re
import json
from uuid import uuid4
import pandas as pd
import numpy as np
from models.classifiers import FastAnswerClassifier
from utils.config import (text_storage,
                          root)
from utils.data_types import Query
from utils.utils import queries2etalon
from datetime import datetime

sys_file_names = [("sys1_train_dataset.csv", "sys1_test_dataset.csv")]
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M")
score = 0.8
sys_id = 1
parameters = {"timestamp": date_time, "score": score, "SysID": sys_id}
if not os.path.exists(os.path.join(root, "data", "results", date_time)):
    os.mkdir(os.path.join(root, "data", "results", date_time))

#  Query: "Query", "templateId, etalonText, etalonId, SysID, moduleId, pubsList"
for train_file_name, test_file_name in sys_file_names:
    text_storage.delete_all_from_table("answers")
    text_storage.delete_all_from_table("etalons")
    text_storage.delete_all_from_table("stopwords")
    test_df = pd.read_csv(os.path.join(root, "data", "datasets", test_file_name), sep="\t")
    worker = FastAnswerClassifier()
    et_data = pd.read_csv(os.path.join(root, "data", "datasets", train_file_name), sep="\t")
    test_df["Cluster"] = test_df["Cluster"].str.replace(r"\s+", " ")

    queries = [Query(tmp, cls, 0, sys_id, 85, "[0]") for tmp, cls in zip(et_data["ID"], et_data["Cluster"])]
    parameters["etalons_number"] = len(queries)
    worker.add_etalons(queries2etalon(queries))
    test_texts = [re.sub(r"\s+", " ", tx) for tx in list(test_df["Cluster"])]
    parameters["test_texts_number"] = len(test_texts)

    searching_results = worker.texts_classification(test_texts, score)
    test_ids, etalons_ids, scores = zip(*searching_results)
    searched_test_texts = [test_texts[i] for i in test_ids]

    searching_results_df = pd.DataFrame(searching_results, columns=["test_id", "etalon_id", "score"])
    searched_test_texts_df = pd.DataFrame(set([(i, test_texts[i]) for i in test_ids]), columns=["test_id", "test_text"])
    searching_results_df_ = pd.merge(searching_results_df, searched_test_texts_df, on="test_id")

    searched_data = text_storage.search_return_all_col(etalons_ids, column_name="etalonId", table_name="etalons")
    searched_data_df = pd.DataFrame([(x[0], x[1], x[2]) for x in searched_data], columns=["predict_id",
                                                                                          "etalon_text", "etalon_id"])

    results_df = pd.merge(searching_results_df_, searched_data_df, on="etalon_id")

    file_name = "".join([date_time, "_SysId", str(sys_id), "_test_results_Jaccard.csv"])
    parameters["result_file_name"] = file_name
    results_df.to_csv(os.path.join(root, "data", "results", date_time, file_name), sep="\t", index=False)

    id_score_max = results_df.groupby("test_id", as_index=False)["score"].max()

    result_dfs = []
    results_df_cut = results_df.drop(["etalon_id", "etalon_text"], axis=1)
    for i, s in zip(id_score_max["test_id"], id_score_max["score"]):
        temp_df = results_df_cut[(results_df_cut["test_id"] == i) & (results_df_cut["score"] == s)]
        result_dfs.append(temp_df)

    results_df_cut = pd.concat(result_dfs, axis=0)
    results_df_cut.drop_duplicates(inplace=True)
    file_name = "".join([date_time, "_SysId", str(sys_id), "_test_results_Jaccard_cut.csv"])
    parameters["result_file_name_cut"] = file_name
    results_df_cut = pd.merge(results_df_cut, test_df, right_on="Cluster", left_on="test_text", how="left")
    results_df_cut["validate"] = np.where((results_df_cut["templateId"] == results_df_cut["predict_id"]), 1, 0)
    results_df_cut.to_csv(os.path.join(root, "data", "results", date_time, file_name), sep="\t", index=False)
    parameters_file_name = "".join([date_time, "_SysID", str(sys_id), "_parameters.json"])
    with open(os.path.join(root, "data", "results", date_time, parameters_file_name), "w") as j_f:
        json.dump(parameters, j_f)

    all_examples = test_df.shape[0]
    scores_ = [0.8, 0.9, 0.95, 0.99]
    test_results_dict = []
    for score in scores_:
        results_sc_df = results_df_cut[results_df_cut["score"] >= score]
        all_examples_sc = results_sc_df.shape[0]
        trues = sum(results_sc_df["validate"])
        false = all_examples_sc - trues
        precision = trues / all_examples_sc
        recall = (trues + false) / all_examples
        test_results_dict.append({"score": score, "precision": round(precision, 2), "recall": round(recall, 2)})

    result_json_name = "".join([date_time, "_test_results.json"])
    with open(os.path.join(root, "data", "results", date_time, result_json_name), "w") as r_j_f:
        json.dump(test_results_dict, r_j_f)