import pandas as pd
import pickle
import numpy as np
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

t = Okt()

raw_train_data = pd.read_csv(
    "utils/training_setences_tmp.csv",
    encoding="utf-8",
)


def cleaning_data(df):
    # 영문자 포함여부 확인
    # white space 데이터를 empty value로 변경
    df["sentences"] = df["sentences"].str.replace("^ +", "", regex=True)
    df["sentences"].replace("", np.nan, inplace=True)
    # print(df.isnull().sum())

    # score값 변경 (string으로 변경)
    # mask = ["A_da", "B_su", "C_co"]
    # df[mask] = df[mask].astype(str)
    return df


raw_train_data = cleaning_data(raw_train_data)


def twit_tokenizer(text):
    target_tags = ["Noun", "Verb", "Adjective"]
    result = []
    for word, tag in t.pos(text, norm=True, stem=True):
        if tag in target_tags:
            result.append(word)
    return result


X = np.array(raw_train_data["sentences"])  # 학습 데이터에서 문장

vect_tmp = TfidfVectorizer(tokenizer=twit_tokenizer)
vec_obj_tmp = vect_tmp.fit(X)
# max_ft = vec_obj_tmp.vocabulary_.__len__() - 26  # 654
# max_ft = vec_obj_tmp.vocabulary_.__len__() - 255
# max_ft = 429 # 그냥... 강제로 지정함.....
# max_ft = 432 # 그냥... 강제로 지정함.....
# max_ft = 1600

# max_ft = 400
max_ft = 40

vectorizer = TfidfVectorizer(
    tokenizer=twit_tokenizer,
    max_features=max_ft,
    ngram_range=(1, 2),
    min_df=2,
)

vec_obj = vectorizer.fit_transform(X)  # train set 을 변환
# print("="*10, vec_obj.transform(X).shape)
# print("trainig text의 vocab단어 수:", vec_obj.vocabulary_.__len__())  # vocab 사이즈 확인
# print("="*10, vec_obj.shape)

# 모델 불러오기
# model0 = pickle.load(open("utils/model/clf_model0.pickle", "rb"))
# model1 = pickle.load(open("utils/model/clf_model1.pickle", "rb"))
# model2 = pickle.load(open("utils/model/clf_model2.pickle", "rb"))
# model3 = pickle.load(open("utils/model/clf_model3.pickle", "rb"))
# model4 = pickle.load(open("utils/model/clf_model4.pickle", "rb"))
# model5 = pickle.load(open("utils/model/clf_model5.pickle", "rb"))


# model0 = pickle.load(open("utils/model/0105_clf_model0.pickle", "rb"))
# model1 = pickle.load(open("utils/model/0105_clf_model1.pickle", "rb"))
# model2 = pickle.load(open("utils/model/0105_clf_model2.pickle", "rb"))
# model3 = pickle.load(open("utils/model/0105_clf_model3.pickle", "rb"))
# model4 = pickle.load(open("utils/model/0105_clf_model4.pickle", "rb"))
# model5 = pickle.load(open("utils/model/0105_clf_model5.pickle", "rb"))

# model0 = pickle.load(open("utils/model/0111_clf_model0.pickle", "rb"))
# model1 = pickle.load(open("utils/model/0111_clf_model1.pickle", "rb"))
# model2 = pickle.load(open("utils/model/0111_clf_model2.pickle", "rb"))
# model3 = pickle.load(open("utils/model/0111_clf_model3.pickle", "rb"))
# model4 = pickle.load(open("utils/model/0111_clf_model4.pickle", "rb"))
# model5 = pickle.load(open("utils/model/0111_clf_model5.pickle", "rb"))

model0 = pickle.load(open("utils/model/0111-2_clf_model0.pickle", "rb"))
model1 = pickle.load(open("utils/model/0111-2_clf_model1.pickle", "rb"))
model2 = pickle.load(open("utils/model/0111-2_clf_model2.pickle", "rb"))
model3 = pickle.load(open("utils/model/0111-2_clf_model3.pickle", "rb"))
model4 = pickle.load(open("utils/model/0111-2_clf_model4.pickle", "rb"))
model5 = pickle.load(open("utils/model/0111-2_clf_model5.pickle", "rb"))

def risk_test(text):
    text_tfidf = vectorizer.transform(np.array([text]))  # test set 을 변환
    scoring_result = {}

    scoring_result["자살 계획의 구체성"] = model0.predict(text_tfidf)
    scoring_result["자살 시도력"] = model1.predict(text_tfidf)
    scoring_result["정신건강의 문제"] = model2.predict(text_tfidf)
    scoring_result["술, 약물 복용 상태"] = model3.predict(text_tfidf)
    scoring_result["지지체계/고립정도/대인관계"] = model4.predict(text_tfidf)
    scoring_result["협조능력"] = model5.predict(text_tfidf)
    
    logging.debug(scoring_result)

    return scoring_result


def classify_suicide_risks(fulltext: list[str]):
    """
    자살 위험성 구분

    risk = {
        "risk_label": risk_label,
        "score": score,
    }
    """

    global scoring_result
    scoring_result = []  # 대화 시작!
    score = 0
    score0 = 0
    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0
    score5 = 0

    logging.debug("fulltext :%s", fulltext)

    for i in range(len(fulltext)):
        text_input = fulltext[i]

        # score = risk_test(text_input).item()
        score0 += risk_test(text_input)["자살 계획의 구체성"][0]
        score1 += risk_test(text_input)["자살 시도력"][0]
        score2 += risk_test(text_input)["정신건강의 문제"][0]
        score3 += risk_test(text_input)["술, 약물 복용 상태"][0]
        score4 += risk_test(text_input)["지지체계/고립정도/대인관계"][0]
        score5 += risk_test(text_input)["협조능력"][0]

    # score = score0[0] + score1[0] + score2[0] + score3[0] + score4[0] + score5[0]
    score = score0 + score1 + score2 + score3 + score4 + score5
    logging.debug("score0 :%s", score0)
    logging.debug("score1 :%s", score1)
    logging.debug("score2 :%s", score2)
    logging.debug("score3 :%s", score3)
    logging.debug("score4 :%s", score4)
    logging.debug("score5 :%s", score5)
    logging.debug("score :%s, type:%s", score, type(score))

    # 점수의 총합
    # if score > 30:
    #     risk_label = "고위험"
    # elif score > 20:
    #     risk_label = "중위험"
    # elif score > 2:
    #     risk_label = "저위험"
    # else:
    #     risk_label = "위험하지 않음"
    
    if score >= 10:
        risk_label = "고위험"
    elif score >= 7:
        risk_label = "중위험"
    elif score < 6:
        risk_label = "저위험"
    else:
        risk_label = "위험하지 않음"

    return {
        "risk_label": risk_label,
        "score": int(score),
        "score_detail": {
            "suicide_plan": int(score0),
            "suicide_attempt": int(score1),
            "mental_problem": int(score2),
            "alcohol_drugs": int(score3),
            "support_system": int(score4),
            "cooperation": int(score5),
        },
    }
