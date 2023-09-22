import itertools
import logging
import re
from konlpy.tag import Kkma
import pandas as pd
from utils.database import (
    database,
    keyword_table,
)

kkma = Kkma()
keyword_db = None


# DB에서 핵심어 사전 불러오기
async def prepare_keyword_dict() -> None:
    query = keyword_table.select()
    rows = await database.fetch_all(query=query)

    df = pd.DataFrame(rows)

    global keyword_db
    keyword_db = df[["id", "purpose", "word", "class_idx", "class", "morphs"]]
    keyword_db = keyword_db.query(
        "purpose != '음성인식'").reset_index(drop=True).dropna()

    # keyword_db["morphs"]의 값들은 전체가 한 덩어리의 string으로 인식되므로, split 함수를 통해 리스트로 만들어야 함
    for i in range(len(keyword_db)):
        keyword_db["morphs"][i] = keyword_db["morphs"][i].split(",")

    # logging.debug(keyword_db)


def pos_filter(target: list):
    # 핵심 품사 목록 리스트
    pos_list = ["NNG", "NNP", "VA", "VV", "OL", "VX",
                "XR", "NP", "NR", "MDN", "NNM", "MAG", "XSN"]
    result = []
    for index, target_item in enumerate(target):
        # target_item이 pos_list에 있으면, 해당되는 target_item(품사) 직전의 target_item(단어)과
        # 품사 target_item을 result에 추가
        if target_item in pos_list:
            result.append(target[index-1])
            result.append(target_item)

    return result


# def lists_intersection(target1: list, target2: list):
#     return [x for x in target1 if x in target2]
def lists_intersection(target1, target2):
    return set(target1) & set(target2)



def extract_keywords(text: str) -> dict:
    # logging.debug("text:", text)
    text_pos = list(itertools.chain(*kkma.pos(text)))
    text_pos = pos_filter(text_pos)
    # logging.debug("filtered text_pos :", text_pos)

    result = {
        "who": [],
        "when": [],
        "where": [],
        "what": [],
        "why": [],
        "how": [],
        "drinking": [],
        "quantity": [],
        "alcohol": [],
        "suicide_method": [],
        "mood": [],
        "mhcenter": [],
        "mental_illness": [],
        "mental_symptom": [],
        "physical_pain": [],
        "crime": [],
        "drugs": [],
    }

    for ss in text.split():
        logging.debug("ss: %s", ss)
        temp = {}
        ktemp = None
        ss_pos = list(itertools.chain(*kkma.pos(ss)))
        ss_pos = pos_filter(ss_pos)
        logging.debug("ss_pos: %s", ss_pos)

        for i, j, k in zip(keyword_db["word"], keyword_db["morphs"],
                           keyword_db["class_idx"]):

            # 단어가 원형 그대로 추출되는 경우
            if (i in ss) and (i in text_pos):
                ktemp = k
                temp["word_segment"] = ss
                temp["keyword"] = i
                temp["start"] = [m.start() for m in re.finditer(ss, text)][0]
                temp["end"] = [m.end() for m in re.finditer(ss, text)][0]
                logging.debug("원형 핵심어 i: %s, word_segment: %s, j: %s, k: %s", i, ss, j, k)
                break

            # 복합 명사 단어
            elif (i in ss) and (i not in text_pos) and (len(ss) > 1) and (len(i) > 1) and \
                {(lists_intersection(j[0::2], text_pos[0::2]) == lists_intersection(j[0::2], ss_pos[0::2])) or
                 (lists_intersection(j, text_pos) == j) or
                 (lists_intersection(j, ss_pos) == j)}:
                logging.debug("t가 띄어쓰기로 쪼개지지 않는 두 음절 이상 단어들의 조합일 때 i: %s, j: %s", i, j)
                # 못살게 굴다, 벌+개탄, 그 전, 애들,

                if len(lists_intersection(j[0::2], text_pos[0::2])) == len(j)/2:
                    logging.debug("!!!!", lists_intersection(j[0::2], text_pos[0::2]))
                    end = len(lists_intersection(j[0::2], text_pos[0::2]))
                    logging.debug("end:%s", end)
                    ktemp = k
                    start = [m.start() for m in re.finditer(ss, text)][0]
                    logging.debug("start:%s", start)
                    temp["word_segment"] = i
                    temp["keyword"] = i
                    temp["start"] = [m.start()
                                     for m in re.finditer(ss, text)][0]
                    temp["end"] = start+len(i)
                    logging.debug("명사형 핵심어1 i: %s, word_segment: %s, j: %s, k: %s", temp["keyword"], ss, j, k)
                    break

                elif ss_pos[0:4] == j[0:4]:
                    ktemp = k
                    temp["word_segment"] = ss
                    temp["keyword"] = i
                    temp["start"] = [m.start()
                                     for m in re.finditer(ss, text)][0]
                    temp["end"] = [m.end() for m in re.finditer(ss, text)][0]
                    logging.debug("명사형 핵심어2 i: %s, word_segment: %s, j: %s, k: %s", temp["keyword"], ss, j, k)
                    break

            elif (len(i.split()) >= 2) and (len(ss) == 1) and (i not in ss) and (ss == i[0])\
                and {(len(lists_intersection(ss_pos[0::2], j[0::2])) > 0) or
                     (len(lists_intersection(text_pos[0::2], j[0::2])) > 0)}:
                logging.debug("띄어쓰기로 쪼개진 한 음절 단어들의 조합일 때 i: %s, j: %s", i, j)
                # 그 전,
                
                if (len(i) >= 3) and (i[2] in lists_intersection(j[0::2], text_pos[0::2])):
                    logging.debug("i[2] : %s", i[2])
                    logging.debug("lists_intersection(j[0::2], text_pos[0::2]) : %s", lists_intersection(j[0::2], text_pos[0::2]))
                    logging.debug("!!!!", lists_intersection(j[0::2], text_pos[0::2]))
                    end = len(lists_intersection(j[0::2], text_pos[0::2])) + 1
                    logging.debug("end:%s", end)
                    ktemp = k
                    start = [m.start() for m in re.finditer(ss, text)][0]

                    temp["word_segment"] = text[start: start+len(ss)+end+1]
                    temp["keyword"] = i
                    temp["start"] = [m.start()
                                     for m in re.finditer(ss, text)][0]
                    temp["end"] = start+len(ss)+end+1
                    logging.debug("비원형 핵심어1 i: %s, word_segment:%s, j:%s, k:%s", temp["keyword"], ss,  j, k)
                    break

            elif (len(i.split()) >= 2) and (len(ss) > 1) and (i not in ss) and \
                {(lists_intersection(j, text_pos) == j) or
                 (lists_intersection(j, lists_intersection(j, text_pos)) == j)}:
                logging.debug("띄어쓰기로 쪼개진 두 음절 이상 단어들의 조합일 때 i:%s, j: %s", i, j)
                # 못살게 굴다,

                if (len(lists_intersection(j[0::2], text_pos[0::2])) == len(j)/2) and\
                    (ss[0] == i[0]) and (ss_pos[0] in j):
                    logging.debug("!!!!", lists_intersection(j[0::2], text_pos[0::2]))
                    end = len(lists_intersection(j[0::2], text_pos[0::2])) + 1
                    logging.debug("end:%s", end)
                    ktemp = k
                    start = [m.start() for m in re.finditer(ss, text)][0]
                    logging.debug("start:%s", start)

                    temp["word_segment"] = text[start: start+len(ss)+end+1]
                    temp["keyword"] = i
                    temp["start"] = [m.start()
                                     for m in re.finditer(ss, text)][0]
                    temp["end"] = start+len(ss)+end+1
                    logging.debug("비원형 핵심어2 i:%s, word_segment:%s, j:%s, k:%s", temp["keyword"], ss, j, k)
                    break

            # 하나의 품사로 추출되지만, 동사/형용사여서 원형 그대로 추출되지 않고
            elif (len(ss) > 1) and (len(i.split()) == 1) and (i not in ss) and (ss[0] == i[0]) and\
                len(lists_intersection(ss_pos[0::2], j[0::2])) > 0 and\
                {(lists_intersection(j, text_pos) == j) or
                 (lists_intersection(j, lists_intersection(j, text_pos)) == j)}:
                logging.debug("i:%s, j:%s", i, j)

                if (len(lists_intersection(j[0::2], text_pos[0::2])) == len(j)/2) and \
                    (ss_pos[1] == j[1]):
                    ktemp = k
                    start = [m.start() for m in re.finditer(ss, text)][0]
                    end = [m.end() for m in re.finditer(ss, text)][0]

                    logging.debug("end:%s", end)
                    # temp["word_segment"] = text[start:end+1]
                    temp["word_segment"] = text[start:end]
                    temp["keyword"] = i
                    temp["start"] = [m.start()
                                     for m in re.finditer(ss, text)][0]
                    temp["end"] = [m.end() for m in re.finditer(ss, text)][0]
                    logging.debug("비원형 핵심어3 i:%s, word_segment:%s, j:%s, k:%s", temp["keyword"], ss, j, k)
                    break

        logging.debug('temp:%s', temp)
        logging.debug("=================================")

        # ktemp 번호에 따른 result의 순서가 정해져 있으므로
        for n in range(1, 18):
            if ktemp == n:
                result[f'{list(result.keys())[n-1]}'].append(temp)

    logging.debug("=================================")
    return result


