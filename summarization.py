import pandas as pd
import re
import math
import numpy as np
from sklearn.preprocessing import normalize


def text_summarizer(fulltext: str):
    # 정규표현식을 이용하여 텍스트를 마침표(.), 물음표(?), 느낌표(!) 중 하나의 문자 뒤에 공백이 하나 이상인 패턴에 따라 문장으로 분리
    sentences = re.split("[\.?!]\s", fulltext)

    data = []

    for sentence in sentences:
        # 빈 문장인 경우 다음 반복으로 이동
        if (sentence == "") or (len(sentence) == 0):
            continue
        temp_dict = dict()
        temp_dict["sentence"] = sentence
        temp_dict["token_list"] = sentence.split()  # 띄어쓰기 단위로 토큰 분리

        data.append(temp_dict)

    df = pd.DataFrame(data)

    # 문장들 간 유사도 계산
    # sentence similarity = len(intersection) / log(len(set_a)) + log(len(set_b))

    similarity_matrix = []

    for i, row_i in df.iterrows():
        i_row_vec = []

        for j, row_j in df.iterrows():
            if i == j:
                i_row_vec.append(0.0)
            else:
                intersection = len(
                    set(row_i["token_list"]) & set(row_j["token_list"]))

                log_i = math.log(len(set(row_i["token_list"])))
                log_j = math.log(len(set(row_j["token_list"])))

                # log_i, log_j가 모두 0인 경우, 분모가 0이 되어 오류 발생 -> similarity = 0이라고 직접 선언
                if not log_i + log_j == 0:
                    similarity = intersection / (log_i + log_j)
                else:
                    similarity = 0.0
                i_row_vec.append(similarity)

        similarity_matrix.append(i_row_vec)

        weighted_graph = np.array(similarity_matrix)

    # 테스트 용이어서 max_iter가 30. 원래대로라면 특정값이 나올 때까지 무한루프
    def pagerank(x, df=0.85, max_iter=50):
        # df 값이 0보다 크고 1보다 작은 범위에 속하지 않으면 AssertionError를 발생
        assert 0 < df < 1

        # initialize
        # normalize 는 pagerank 알고리즘의 페이지 중요도를 다 더해서 1로 만들어주는 작업
        A = normalize(x, axis=0, norm="l1")

        # R은 각 문장의 rank값을 의미하고 맨처음은 1로 초기화
        R = np.ones(A.shape[0]).reshape(-1, 1)

        # bias는 만족하지 못하고 페이지를 떠나는 확률로 (1 - damping factor)를 의미
        # 각 rank는 weighted graph * rank*damping factor + (1-damping factor)로 계산
        bias = (1 - df) * R

        # iteration
        # '_'를 변수 대신 사용하여 반복 횟수를 저장하지 않음
        # 원래 pagerank에서는 각 랭크 값이 어느정도 수렴할때까지 무한 루프를 돌리는데 이건 간단한 테스트이기 때문에 max iteration 만큼 동작
        for _ in range(max_iter):
            R = df * (A * R) + bias

        return R

    # pagerank를 통해 rank matrix 반환
    R = pagerank(weighted_graph)

    # 반환된 matrix를 row 별로 sum
    R = R.sum(axis=1)

    # 해당 rank 값을 sort, 값이 높은 n개의 문장 index를 반환
    summary_num = 5
    pre_indexs = R.argsort()
    if len(pre_indexs) > summary_num:
        indexs = R.argsort()[-summary_num:]
    else:
        indexs = R.argsort()

    # rank 값이 높은 문장을 high_ranks 리스트에 담기
    high_ranks = [df["sentence"][index] for index in sorted(indexs)]

    # high_ranks 리스트에 담긴 문장들을 한 문자열로 취급하기 위해 for문 사용
    summarized_text = ""

    for rank in high_ranks:
        # 간혹 마침표 없는 문장의 경우 마침표를 더해줌
        if "." not in rank:
            rank += "."

        # 마지막 문장인 경우에는 문장 뒤에 띄어쓰기 x
        if rank != high_ranks[-1]:
            rank += " "

        summarized_text += rank

    summarized_text

    return summarized_text
