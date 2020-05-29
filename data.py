import random
from typing import List
import numpy as np
from scipy import spatial


class Document:
    def __init__(self, id: int, attributes: list, rel_score: float,
                 knn=None, subprofile=None, feature_vec=None):
        if knn is None:
            knn = []
        if subprofile is None:
            subprofile = set()

        self.id: int = id
        self.attributes = set(attributes)
        self.rel_score = rel_score
        self.knn: List[int] = knn
        self.subprofile = subprofile
        self.feature_vec = feature_vec

    def update_knn(self, knn):
        self.knn = knn

    def __str__(self):
        return "Document(id=%d, attributes=[%s], rel_score=%.3f, knn=[%s])" % (self.id,
                                                                               ", ".join(map(str, self.attributes)),
                                                                               self.rel_score,
                                                                               ", ".join(map(str, self.knn)))


def get_fake_docs(N=100, num_genres=20):
    """
    generate 100 documents with less than 5 attributes each, and attributes are integer ids in range [0, 20]
    relevant score for each document is generated randomly in range [0, 1)
    knn's are calculated by jaccard sim with attributes
    :return:
    """
    random.seed(0)  # to give static data
    docs = []
    for i in range(N):
        id = i
        attr = []
        for j in range(5):
            rand_attr = random.randint(0, num_genres * 2)
            if rand_attr <= num_genres:
                attr.append(rand_attr)
        if len(attr) == 0:
            attr.append(random.randint(0, num_genres))

        rel_score = random.random()

        docs.append(Document(id, attr, rel_score))

    update_knn_by_jaccard_sim(docs, k=20)

    return docs


def jaccard_similarity(a: Document, b: Document):
    a_set = set(a.attributes)
    b_set = set(b.attributes)
    inter_set = a_set.intersection(b_set)

    return len(inter_set) / (len(a_set) + len(b_set) - len(inter_set))


def cosine_similarity(a: Document, b: Document):
    return 1 - spatial.distance.cosine(a.feature_vec, b.feature_vec)


def update_knn_by_jaccard_sim(docs: List[Document], k=5) -> None:
    N = len(docs)
    sim_mat = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if i > j:
                sim_mat[i, j] = sim_mat[j, i]
            else:
                sim_mat[i, j] = jaccard_similarity(docs[i], docs[j])
    # print(sim_mat)
    for i in range(N):
        knn = np.argsort(sim_mat[i])[::-1][:k + 1].tolist()
        knn = list(filter(lambda id: id != i, knn))
        docs[i].update_knn(knn[:k])


def get_session(K=100, N=100):
    """
    Get a K length recommendation set and a N length User History
    :param K:
    :param N:
    :return:
    """
    docs = get_fake_docs(K + N + 500)
    RS = docs[:K]
    I_u = docs[K:]
    return RS, I_u


if __name__ == "__main__":
    docs = get_fake_docs(100)
    for doc in docs:
        print(doc)
