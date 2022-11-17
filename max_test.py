import os
import pandas as pd
from utils.config import root

df = pd.read_csv(os.path.join(root, "data", "results", "20221117_1239",
                              "20221117_1239_SysId1_test_results_Jaccard.csv"), sep="\t")

# results_df = df[["test_id", "test_text", "answer_id", "score"]].groupby("score").max()
results_df = df.groupby("score").max()
results_df.to_csv("results_df.csv", sep="\t")


# results_df = df.groupby(["test_text", "test_id"], as_index=False)["score"].max()
print(df[df["test_id"] == 35])
df_ = df[df["test_id"] == 35].groupby("test_id", as_index=False)["score"].max()
print(df_)
print(type(df_))
print(df_["score"].values[0])
print(df[(df["test_id"] == 35) & (df["score"] == df_["score"].values[0])])
df_test = df[(df["test_id"] == 35) & (df["score"] == df_["score"].values[0])]
df_test.to_csv("df_test.csv", sep="\t")
id_score_max = df.groupby("test_id", as_index=False)["score"].max()

print(id_score_max )
df_list = []
df.drop(["etalon_id", "etalon_text"], axis=1, inplace=True)
for i, s in zip(id_score_max["test_id"], id_score_max["score"]):
    temp_df = df[(df["test_id"] == i) & (df["score"] == s)]
    df_list.append(temp_df)

result_df = pd.concat(df_list, axis=0)
print(result_df)
result_df.to_csv("result_df.csv", sep="\t")


# print(df["answer_id"][df["test_id"] == 35]["score"].max())
# print(df.groupby("answer_id", as_index=False)["score"].max())

# id_score_max = df.groupby("test_id", as_index=False)["score"].max()
# result_df = pd.merge(df, id_score_max, on="test_id")
# result_df.to_csv("result_df.csv", sep="\t")
# print(result_df)
# temp_df = df.drop(["etalon_id", "etalon_text"], axis=1)[df["test_id"] == 35].groupby("score").max()
# temp_df.to_csv("temp_df.csv", sep="\t")

'''
result_dfs = []
for num, t_id in enumerate(set(df["test_id"])):
    temp_df = df.drop(["etalon_id", "etalon_text"], axis=1)[df["test_id"] == t_id].groupby("score", as_index=False).max()
    # df_temp = df[df["test_id"] == t_id].groupby("score").max()
    result_dfs.append(temp_df)
    print(num, "/", len(set(df["test_id"])))

results_df = pd.concat(result_dfs, axis=0)
print(results_df)
results_df.to_csv("results_df.csv", sep="\t")
'''
'''    
results_df = df.groupby(["test_text", "test_id", "answer_id"], as_index=False)["score"].max()
print(results_df)
print(results_df[results_df["test_id"] == 35])
'''
# 2417 rows x 3 columns