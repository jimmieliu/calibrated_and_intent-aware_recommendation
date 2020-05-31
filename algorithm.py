#!/bin/python3
from __future__ import division
from data import get_session
from data import Document
from typing import List, Set
from subprofile import get_subprofiles
from metrics import summary_item_list, ILD
import math

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
        ts = time.time()
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

        useful_S_index_set = set()
        sum_rel_score_of_S_index = {}
        for item in RS:
            for S_index in range(len(self.S_u_KNN)):
                if item.id in self.S_u_KNN[S_index]:
                    useful_S_index_set.add(S_index)
                    sum_rel_score_of_S_index[S_index] = sum_rel_score_of_S_index.get(S_index, 0.) + item.rel_score

        self.sum_rel_score_of_S_index = sum_rel_score_of_S_index
        self.useful_S_index_set = useful_S_index_set
        print("useful_S_index_set size = ", len(useful_S_index_set))

        super(SPADReranker, self).__init__(RS, I_u)
        print("SPAD init used", (time.time() - ts) * 1000, "ms")

    def P_S_given_u(self, S_index):
        S = self.S_u[S_index]
        return len(S) / self.P_S_given_u_denonimator

    def P_i_given_u_S(self, item: Document, S_index):
        if item.id not in self.S_u_KNN[S_index]:
            return 0.
        else:
            return item.rel_score / self.sum_rel_score_of_S_index[S_index]

    def div_or_cal_fn(self, item: Document):  # div_IA_i_RL
        div_IA = 0.
        for S_index in self.useful_S_index_set:
            tmp_ = self.P_S_given_u(S_index) * self.P_i_given_u_S(item, S_index)
            for item_j in self.RL:
                tmp_ *= (1 - self.P_i_given_u_S(item_j, S_index))

            div_IA += tmp_
        return div_IA


class CRfReranker(Reranker):
    def __init__(self, RS, I_u):
        self.alpha = 0.01
        self.FEATURE_SET = list(range(NUM_GENRES + 1))

        P_f_given_u_denominator = sum([item.w_u_i for item in I_u])
        P_f_gven_u_numerators = {}
        P_f_given_i = {}

        self.P_f_given_u_denominator = P_f_given_u_denominator

        for item in I_u + RS:
            if item.id not in P_f_given_i:
                P_f_given_i[item.id] = {}

            for f in item.attributes:
                P_f_given_i[item.id][f] = 1 / len(item.attributes)

        self.P_f_given_i = P_f_given_i

        for f in self.FEATURE_SET:
            P_f_gven_u_numerators[f] = sum([P_f_given_i[item.id].get(f, 0) for item in I_u])
        self.P_f_gven_u_numerators = P_f_gven_u_numerators

        self.P_f_given_u = {f: self._P_f_given_u(f) for f in self.FEATURE_SET}

        super(CRfReranker, self).__init__(RS, I_u)

    def _P_f_given_u(self, f):
        return self.P_f_gven_u_numerators[f] / self.P_f_given_u_denominator

    def q_f_given_u(self, f, RL):
        if len(RL) == 0:
            return 0.
        return sum([item.w_u_i * self.P_f_given_i[item.id].get(f, 0) for item in RL]) \
               / sum([item.w_u_i for item in RL])

    def q_f_given_u_tilde(self, f, P_f_given_u, RL):
        return (1 - self.alpha) * self.q_f_given_u(f, RL) + self.alpha * P_f_given_u

    def C_KL_q_p(self, RL):
        C_KL = 0
        for f in self.FEATURE_SET:
            P_f_given_u = self.P_f_given_u[f]
            C_KL += P_f_given_u * \
                    math.log(P_f_given_u / self.q_f_given_u_tilde(f, P_f_given_u, RL))
        return C_KL

    def div_or_cal_fn(self, item):
        return self.C_KL_q_p(self.RL) - self.C_KL_q_p([item] + self.RL)


class CRsReranker(Reranker):
    def __init__(self, RS, I_u):
        self.alpha = 0.01
        self.useful_S_index = set()
        RS_id_item = set()
        id2item = {}
        for item in RS:
            id2item[item.id] = item
            RS_id_item.add(item.id)
        for item in I_u:
            id2item[item.id] = item

        subprofiles = get_subprofiles(I_u)

        P_S_given_u_denominator = sum([item.w_u_i for item in I_u])
        P_S_gven_u_numerators = {}
        P_S_given_i = {}

        self.P_S_given_u_denominator = P_S_given_u_denominator

        self.S_u_KNN = []
        for S_index in range(len(subprofiles)):
            knn_set = set()
            S = subprofiles[S_index]
            for _id in S:
                knn_set.update(id2item[_id].knn)
            self.S_u_KNN.append(knn_set)

        for S_index in range(len(self.S_u_KNN)):
            S = self.S_u_KNN[S_index]
            for _id in S:
                if _id in id2item:
                    if _id in RS_id_item:
                        self.useful_S_index.add(S_index)
                    P_S_given_i[_id] = P_S_given_i.get(_id, 0) + 1

        for _id in P_S_given_i:
            P_S_given_i[_id] = 1 / P_S_given_i[_id]

        self.P_S_given_i = P_S_given_i

        for S_index in range(len(self.S_u_KNN)):
            P_S_gven_u_numerators[S_index] = \
                sum([P_S_given_i[item.id] for item in I_u if item.id in self.S_u_KNN[S_index]])
        self.P_S_gven_u_numerators = P_S_gven_u_numerators

        self.P_S_given_u = {S_index: self._P_S_given_u(S_index) for S_index in range(len(self.S_u_KNN))}

        print("useful_S_index", len(self.useful_S_index))
        super(CRsReranker, self).__init__(RS, I_u)

    def _P_S_given_u(self, S_index):
        return self.P_S_gven_u_numerators[S_index] / self.P_S_given_u_denominator

    def q_S_given_u(self, S_index, RL):
        if len(RL) == 0:
            return 0.
        return sum([item.w_u_i * self.P_S_given_i.get(item.id, 0) for item in RL]) \
               / sum([item.w_u_i for item in RL])

    def q_S_given_u_tilde(self, S_index, P_S_given_u, RL):
        return (1 - self.alpha) * self.q_S_given_u(S_index, RL) + self.alpha * P_S_given_u

    def C_KL_q_p(self, RL):
        C_KL = 0
        for S_index in self.useful_S_index:
            P_S_given_u = self.P_S_given_u[S_index]
            C_KL += P_S_given_u * \
                    math.log(P_S_given_u / self.q_S_given_u_tilde(S_index, P_S_given_u, RL))
        return C_KL

    def div_or_cal_fn(self, item):
        return self.C_KL_q_p(self.RL) - self.C_KL_q_p([item] + self.RL)


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

    ts = time.time()
    for retry in range(retry_time):
        RL_CRf = CRfReranker(RS, I_u).get_RL()
    print("CRf rerank used:", (time.time() - ts) * 1000 / retry_time, "ms")

    ts = time.time()
    for retry in range(retry_time):
        RL_CRs = CRsReranker(RS, I_u).get_RL()
    print("CRs rerank used:", (time.time() - ts) * 1000 / retry_time, "ms")

    print("User Profile Summary:", ", ".join(summary_item_list(I_u)))
    print("User Plain Summary:", ", ".join(summary_item_list(RL_plain)))
    print("User RL_xQuAD Summary:", ", ".join(summary_item_list(RL_xQuAD)))
    print("User RL_SPAD Summary:", ", ".join(summary_item_list(RL_SPAD)))
    print("User RL_CRf Summary:", ", ".join(summary_item_list(RL_CRf)))
    print("User RL_CRs Summary:", ", ".join(summary_item_list(RL_CRs)))

    print("User RL_plain ILD:", ILD(RL_plain))
    print("User RL_xQuAD ILD:", ILD(RL_xQuAD))
    print("User RL_SPAD ILD:", ILD(RL_SPAD))
    print("User RL_CRf ILD:", ILD(RL_CRf))
    print("User RL_CRs ILD:", ILD(RL_CRs))

    for i in range(RL_SIZE):
        print("\n%d - plain | xQuAD" % i)
        print(RL_plain[i], " | ", RL_xQuAD[i],
              "\n | ", RL_SPAD[i], " | ", RL_CRf[i],
              "\n | ", RL_CRs[i])
