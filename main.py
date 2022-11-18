# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import re
import json
import pandas as pd
from models.classifiers import FastAnswerClassifier
from utils.config import (text_storage,
                          root)
from utils.utils import worker_fill
from utils.data_types import Query
from datetime import datetime

sys_file_names = [(1, "sys1_testing_set.csv"), (2, "sys2_testing_set.csv"), (8, "sys3_testing_set.csv")]
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M")
score = 0.6
parameters = {"timestamp": date_time, "score": score}
os.mkdir(os.path.join(root, "data", "results", date_time))

for sys_id, file_name in sys_file_names:
    parameters["SysID"] = sys_id
    test_df = pd.read_csv(os.path.join(root, "data", file_name), sep="\t")
    fa_worker = FastAnswerClassifier()
    et_data = text_storage.search_return_all_col(ids=[sys_id], column_name="SysID", table_name="etalons")
    pubs_data = [Query(*x) for x in et_data]
    parameters["etalons_number"] = len(pubs_data)

    worker = worker_fill(fa_worker, pubs_data, [])
    test_texts = [re.sub(r"\s+", " ", tx) for tx in list(test_df["text"])]
    parameters["test_texts_number"] = len(test_texts)

    searching_results = worker.texts_classification(test_texts, score)
    test_ids, etalons_ids, scores = zip(*searching_results)
    searched_test_texts = [test_texts[i] for i in test_ids]

    searching_results_df = pd.DataFrame(searching_results, columns=["test_id", "etalon_id", "score"])
    searched_test_texts_df = pd.DataFrame(set([(i, test_texts[i]) for i in test_ids]), columns=["test_id", "test_text"])
    searching_results_df_ = pd.merge(searching_results_df, searched_test_texts_df, on="test_id")

    searched_data = text_storage.search_return_all_col(etalons_ids, column_name="etalonId", table_name="etalons")
    searched_data_df = pd.DataFrame([(x[0], x[1], x[2]) for x in searched_data], columns=["answer_id",
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

    results_df_cut.to_csv(os.path.join(root, "data", "results", date_time, file_name), sep="\t", index=False)
    parameters_file_name = "".join([date_time, "_SysID", str(sys_id), "_parameters.json"])
    with open(os.path.join(root, "data", "results", date_time, parameters_file_name), "w") as j_f:
        json.dump(parameters, j_f)
