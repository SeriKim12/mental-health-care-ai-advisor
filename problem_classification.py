from utils.nlp.keyword_extraction import extract_keywords


def classify_problem(fulltext: list[str]):
    """
    문제유형 분류

    problem = {
        "suicide": 0.9,  # 확률을 0 에서 1 사이 값으로 출력
        "etc": 0.1,
    }
    """

    keyword_list = [extract_keywords(text) for text in fulltext]

    # 초기화
    suicide_word_count = 0
    problem = {
        "suicide": 0.0,
        "etc": 1.0,
    }

    # 전체 문장에 자살 관련 단어가 있으면 카운트
    for keyword in keyword_list:
        if len(keyword["what"]) > 0:
            suicide_word_count += len(keyword["what"])

    # 자살 단어 카운트가 2를 넘으면 자살로 판단
    if suicide_word_count >= 2:
        problem = {
            "suicide": 1.0,
            "etc": 0.0,
        }

    return problem


class ProblemClassifier:
    def __init__(self):
        self.init()

    def init(self):
        self._suicide_word_count = 0

    def get(self):
        if self._suicide_word_count >= 2:
            problem = {
                "suicide": 1.0,
                "etc": 0.0,
            }
        else:
            problem = {
                "suicide": 0.0,
                "etc": 1.0,
            }

        return problem

    def feed(self, keywords):
        if len(keywords["what"]) > 0:
            self._suicide_word_count += len(keywords["what"])

