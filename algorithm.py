#!/bin/python3
from __future__ import division
from data import get_session
from data import Document
from typing import List, Set
from subprofile import get_subprofiles

LAMBDA = 0.7
NUM_GENRES = 20
RL_SIZE = 5


class Reranker:
    def __init__(self, RS: List[Document], I_u: List[Document]):
        """
        :param RS: Recommendation candidates
        :param I_u: Items that are in user's profile ( user click history )
        """
        self.RS = RS
        self.I_u = I_u
        self.RL = []

    def div_or_cal_fn(self, item):
        return item.rel_score

    def f_obj(self, item):
        return (1 - LAMBDA) * item.rel_score + LAMBDA * self.div_or_cal_fn(item)

    def get_RL(self):
        """
        :return: recommendation list
        """
        RS = self.RS
        RL = self.RL

        used_mask = [False] * len(RS)

        while len(RL) < min(RL_SIZE, len(RS)):
            max_item_index = -1
            max_score = float("-inf")
            for i in range(len(RS)):
                if not used_mask[i]:
                    item = RS[i]
                    cur_score = self.f_obj(item)
                    if cur_score > max_score:
                        max_item_index = i
                        max_score = cur_score

            RL.append(RS[max_item_index])
            used_mask[max_item_index] = True

        return RL


class PlainReranker(Reranker):
    def __init__(self, RS: List[Document], I_u: List[Document]):
        super(PlainReranker, self).__init__(RS, I_u)

    def f_obj(self, item):
        return item.rel_score


class xQuADReranker(Reranker):
    def __init__(self, RS: List[Document], I_u: List[Document]):
        self.FEATURE_SET = list(range(NUM_GENRES + 1))

        num_item_of_feature_f = {}
        for item in I_u:
            for f in item.attributes:
                num_item_of_feature_f[f] = num_item_of_feature_f.get(f, 0) + 1

        self.num_item_of_feature_f = num_item_of_feature_f

        self.P_f_given_u_denominator = sum([num_item_of_feature_f[f] for f in num_item_of_feature_f])

        sum_RS_rel_of_feature_f = {}
        for item in RS:
            for f in item.attributes:
                sum_RS_rel_of_feature_f[f] = sum_RS_rel_of_feature_f.get(f, 0) + item.rel_score

        self.sum_RS_rel_of_feature_f = sum_RS_rel_of_feature_f

        super(xQuADReranker, self).__init__(RS, I_u)

    def xQuAD_P_f_given_u(self, f):
        return self.num_item_of_feature_f[f] / self.P_f_given_u_denominator

    def xQuAD_P_i_given_u_f(self, item: Document, f):
        if f in item.attributes:
            return item.rel_score / self.sum_RS_rel_of_feature_f[f]
        else:
            return 0.

    def div_or_cal_fn(self, item: Document):  # div_IA_i_RL
        div_IA = 0.
        for f in self.FEATURE_SET:
            tmp_ = self.xQuAD_P_f_given_u(f) * self.xQuAD_P_i_given_u_f(item, f)
            for item_j in self.RL:
                tmp_ *= (1 - self.xQuAD_P_i_given_u_f(item_j, f))
            div_IA += tmp_
        return div_IA


class SPADReranker(Reranker):
    def __init__(self, RS: List[Document], I_u: List[Document]):
        self.S_u = get_subprofiles(I_u)
        print("S_u size = ", len(self.S_u))
        self.S_u_KNN = []
        id2item = {}
        for item in I_u:
            id2item[item.id] = item

        for S in self.S_u:
            knn_set = set()
            for _id in S:
                knn_set.update(id2item[_id].knn)
            self.S_u_KNN.append(knn_set)

        self.P_S_given_u_denonimator = sum([len(S) for S in self.S_u])

        sum_rel_score_of_S_index = {}
        for item in RS:
            for S_index in range(len(self.S_u_KNN)):
                if item.id in self.S_u_KNN[S_index]:
                    sum_rel_score_of_S_index[S_index] = sum_rel_score_of_S_index.get(S_index, 0.) + item.rel_score

        self.sum_rel_score_of_S_index = sum_rel_score_of_S_index

        super(SPADReranker, self).__init__(RS, I_u)

    def P_S_given_u(self, S_index):
        S = self.S_u[S_index]
        return len(S) / self.P_S_given_u_denonimator

    def P_i_given_u_S(self, item: Document, S_index):
        if item.id not in self.S_u_KNN[S_index]:
            return 0.
        else:
            return item.rel_score / self.sum_rel_score_of_S_index[S_index]

    def div_or_cal_fn(self, item: Document): # div_IA_i_RL
        div_IA = 0.
        for S_index in range(len(self.S_u)):
            tmp_ = self.P_S_given_u(S_index) * self.P_i_given_u_S(item, S_index)
            for item_j in self.RL:
                tmp_ *= (1 - self.P_i_given_u_S(item_j, S_index))

            div_IA += tmp_
        return div_IA


if __name__ == "__main__":
    import time

    RL_SIZE = 5

    RS, I_u = get_session()

    retry_time = 10
    ts = time.time()
    for retry in range(retry_time):
        RL_plain = PlainReranker(RS, I_u).get_RL()
    print("plain rerank used:", (time.time() - ts) * 1000 / retry_time, "ms")

    ts = time.time()
    for retry in range(retry_time):
        RL_xQuAD = xQuADReranker(RS, I_u).get_RL()
    print("xQuAD rerank used:", (time.time() - ts) * 1000 / retry_time, "ms")

    ts = time.time()
    for retry in range(retry_time):
        RL_SPAD = SPADReranker(RS, I_u).get_RL()
    print("SPAD rerank used:", (time.time() - ts) * 1000 / retry_time, "ms")

    for i in range(RL_SIZE):
        print("\n%d - plain | xQuAD" % i)
        print(RL_plain[i], " | ", RL_xQuAD[i], "\n | ", RL_SPAD[i])
