from typing import List
from data import Document
from data import jaccard_similarity


def summary_item_list(item_list: List[Document]):
    feature_cnt = {}
    for item in item_list:
        for f in item.attributes:
            feature_cnt[f] = feature_cnt.get(f, 0) + 1

    feauture_list = [(k, v / len(item_list)) for k, v in feature_cnt.items()]
    feauture_list.sort(key=lambda t: -t[1])

    return ["%d:%.2f" % (k, v) for k, v in feauture_list]


def ILD(item_list: List[Document]):
    sum_ = 0.
    for i in range(len(item_list)):
        item_i = item_list[i]
        for j in range(i + 1, len(item_list)):
            item_j = item_list[j]
            sum_ += jaccard_similarity(item_j, item_i) * 2

    return sum_ * 2 / (len(item_list) * (len(item_list) - 1))


def sum_rel_score(item_list: List[Document]):
    pass
