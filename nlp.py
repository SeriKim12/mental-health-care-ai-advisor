import logging
import sys
from fastapi import APIRouter
from pydantic import BaseModel, Field
import requests
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    pipeline,
)
from utils.nlp.keyword_extraction import extract_keywords
from utils.nlp.problem_classification import classify_problem, ProblemClassifier
from utils.nlp.suicide_risk_classification import classify_suicide_risks
from utils.nlp.suicide_risk_evaluation import SuicideRiskEvaluator

from utils.nlp.summarization import text_summarizer

router = APIRouter()

pipe = None


@router.on_event("startup")
async def startup():
    global pipe
    tokenizer = AutoTokenizer.from_pretrained("alaggung/bart-r3f")
    model = BartForConditionalGeneration.from_pretrained("alaggung/bart-r3f")

    pipe = pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # -1 for CPU, 0 for GPU
    )


# class SummarizationRequest(BaseModel):
#     text: str = Field(
#         ...,
#         description="텍스트",
#         example="안녕하세요. 요약할 텍스트를 입력해주세요.",
#     )


# class SummarizationResponse(BaseModel):
#     summary: str = Field(
#         ...,
#         description="요약된 텍스트",
#         example="안녕하세요. 요약할 텍스트를 입력해주세요.",
#     )


# @router.post(
#     "/summarization/",
#     response_model=SummarizationResponse,
#     summary="텍스트 요약",
# )
# async def summarization(request: SummarizationRequest):
#     """
#     텍스트 요약
#     """

#     labels = pipe(request.text)
#     result = labels[0].get("summary_text") if labels else None
#     return SummarizationResponse(summary=result)


class SummarizationRequest(BaseModel):
    """
    텍스트 요약
    """
    text: str = Field(
        ...,
        description="텍스트",
        example="안녕하세요. 요약할 텍스트를 입력해주세요.",
    )


class SummarizationResponse(BaseModel):
    summary: str = Field(
        ...,
        description="텍스트",
        example="안녕하세요. 요약할 텍스트를 입력해주세요.",
    )


@router.post(
    "/summarization/",
    response_model=SummarizationResponse,
    summary="텍스트 요약",
)
async def summarization(request: SummarizationRequest):
    """
    텍스트 요약 수행
    """
    # return SummarizationResponse(
    #     text_summarizer(request.text))
    summary_result = text_summarizer(request.text)
    return SummarizationResponse(summary=summary_result)


class KeywordExtractionRequest(BaseModel):
    """
    핵심어 추출 요청
    """

    text: str = Field(
        ...,
        description="텍스트",
        example="안녕하세요. 핵심어 추출을 위한 텍스트를 입력해주세요.",
    )


class KeywordExtractionResult(BaseModel):
    """
    핵심어 추출 결과
    """

    word_segment: str = Field(
        ...,
        description="단어 세그먼트",
        example="자살을",
    )
    keyword: str = Field(
        ...,
        description="핵심어",
        example="자살",
    )

    start: int = Field(
        ...,
        description="핵심어 시작 위치",
        example=0,
    )

    end: int = Field(
        ...,
        description="핵심어 끝 위치",
        example=3,
    )


class KeywordExtractionResponse(BaseModel):
    """
    핵심어 추출 응답
    """

    who: list[KeywordExtractionResult] = Field(
        ...,
        description="사람을 지칭하는 단어",
    )

    when: list[KeywordExtractionResult] = Field(
        ...,
        description="시점을 의미하는 단어",
    )

    where: list[KeywordExtractionResult] = Field(
        ...,
        description="장소를 의미하는 단어",
    )

    what: list[KeywordExtractionResult] = Field(
        ...,
        description="내용을 의미하는 단어?",
    )

    why: list[KeywordExtractionResult] = Field(
        ...,
        description="원인을 지칭하는 단어?",
    )

    how: list[KeywordExtractionResult] = Field(
        ...,
        description="방법을 의미하는 단어?",
    )

    drinking: list[KeywordExtractionResult] = Field(
        ...,
        description="음주를 의미하는 단어",
    )

    quantity: list[KeywordExtractionResult] = Field(
        ...,
        description="수량을 지칭하는 단어",
    )

    alcohol: list[KeywordExtractionResult] = Field(
        ...,
        description="술을 지칭하는 단어",
    )

    suicide_method: list[KeywordExtractionResult] = Field(
        ...,
        description="자살수단을 지칭하는 단어",
    )

    mood: list[KeywordExtractionResult] = Field(
        ...,
        description="현재 기분 상태 혹은 전반적인 정서 표현을 지칭하는 단어",
    )

    mhcenter: list[KeywordExtractionResult] = Field(
        ...,
        description="정신건강 관련기관에 해당하는 단어",
    )

    mental_illness: list[KeywordExtractionResult] = Field(
        ...,
        description="정신질환의 병명",
    )

    mental_symptom: list[KeywordExtractionResult] = Field(
        ...,
        description="정신질환의 증상이나 고통을 나타내는 단어",
    )

    physical_pain: list[KeywordExtractionResult] = Field(
        ...,
        description="신체적 질병 혹은 고통",
    )

    crime: list[KeywordExtractionResult] = Field(
        ...,
        description="범죄 혹은 폭력 전반을 아우르는 단어",
    )

    drugs: list[KeywordExtractionResult] = Field(
        ...,
        description="약물 혹은 의약품",
    )


@router.post(
    "/keyword/",
    response_model=KeywordExtractionResponse,
    summary="핵심어 추출",
)
async def get_keywords(request: KeywordExtractionRequest):
    """
    핵심어 추출 수행
    """

    return KeywordExtractionResponse(
        **extract_keywords(request.text)
    )
    # return extract_keywords(request.text)


class ProblemClassificationRequest(BaseModel):
    """
    문제 유형 분류 요청
    """

    fulltext: list[str] = Field(
        ...,
        description="전체 본문",
    )


@router.post(
    "/problem/",
    summary="문제 유형 분류",
)
async def get_problem(request: ProblemClassificationRequest):
    """
    문제 유형 분류 수행
    """

    problem = classify_problem(request.fulltext)

    return problem


class SuicideRiskClassificationRequest(BaseModel):
    """
    자살 위험성 구분 요청
    """

    fulltext: list[str] = Field(
        ...,
        description="전체 본문",
    )


@router.post(
    "/suicide_risks/",
    summary="자살 위험성 구분",
)
async def get_suicide_risks(request: SuicideRiskClassificationRequest):
    """
    자살 위험성 구분 수행
    """

    risks = classify_suicide_risks(request.fulltext)
    print('risks type :', type(risks))

    return risks


# class SentenceNlpRequest(BaseModel):
#     text: str = Field(..., description="텍스트")


# @router.post(
#     "/sentence/"
# )
# async def sentence_nlp(request: SentenceNlpRequest):
#     response = requests.post(
#         "http://localhost:5005/model/parse", json={"text": request.text})
#     intent = response.json().get("intent")
#     intent_name = intent.get("name")
#     return {
#         "keywords": extract_keywords(request.text),
#         "intent": intent_name,
#     }


# class DialogNlpRequest(BaseModel):
#     keywords: list = Field(..., description="키워드")
#     intents: list = Field(..., description="의도")
#     texts: list = Field(...,  description="텍스트")


# @router.post(
#     "/dialog/"
# )
# async def dialog_nlp(request: DialogNlpRequest):
#     problem_classifier = ProblemClassifier()
#     for keyword in request.keywords:
#         problem_classifier.feed(keyword)
#     risk_evaluator = SuicideRiskEvaluator()
#     for intent, text in zip(request.intents, request.texts):
#         risk_evaluator.feed(text, intent)

#     return {
#         "problem": problem_classifier.get(),
#         "risk": risk_evaluator.get(),
#     }


# logging.basicConfig(level=logging.DEBUG,
#                     format='[%(asctime)s][%(levelname)s] %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     stream=sys.stdout, )  # filename = './debug.log')
