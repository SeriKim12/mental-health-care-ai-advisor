from fastapi import APIRouter, Path
from pydantic import BaseModel, Field
from utils.database import (
    database,
    keyword_table,
)

router = APIRouter()


# CRUD 생성 조회 수정 삭제
# 조회
@router.get(
    "/",
    summary="핵심어 사전 조회",
)
async def get_keyword_dict():
    """
    핵심어 사전 조회
    """

    query = keyword_table.select()

    return await database.fetch_all(query)


# 생성
class CreateKeywordRequest(BaseModel):
    """
    핵심어 사전 생성 요청
    """

    purpose: str | None = Field(default="", description="목적")
    word: str = Field(..., description="핵심어")
    class_idx: int | None = Field(default=0, description="클래스 인덱스")
    class_name: str | None = Field(default="", description="클래스")
    morphs: str | None = Field(default="", description="형태소")
    meaning: str | None = Field(default="", description="의미")
    example: str | None = Field(default="", description="예시")
    source: str | None = Field(default="", description="출처")


@router.put(
    "/",
    summary="핵심어 사전 생성",
)
async def create_keyword_dict(request: CreateKeywordRequest):
    """
    핵심어 사전 생성
    """

    query = keyword_table.insert().values(
        purpose=request.purpose,
        word=request.word,
        class_idx=request.class_idx,
        **{"class": request.class_name},
        morphs=request.morphs,
        meaning=request.meaning,
        example=request.example,
        source=request.source,
    )

    # 추가된 단어의 id를 반환
    await database.execute(query)
    return {"message": f"word: {request.word}가 생성되었습니다"}


# 수정
class UpdateKeywordRequest(BaseModel):
    """
    핵심어 사전 수정 요청
    """

    id: int = Field(..., description="ID")
    purpose: str | None = Field(default="", description="목적")
    word: str = Field(..., description="핵심어")
    class_idx: int | None = Field(default=0, description="클래스 인덱스")
    class_name: str | None = Field(default="", description="클래스")
    morphs: str | None = Field(default="", description="형태소")
    meaning: str | None = Field(default="", description="의미")
    example: str | None = Field(default="", description="예시")
    source: str | None = Field(default="", description="출처")


@router.post(
    "/",
    summary="핵심어 사전 수정",
)
async def update_keyword_dict(request: UpdateKeywordRequest):
    """
    핵심어 사전 수정
    """

    query = keyword_table.update().values(
        purpose=request.purpose,
        word=request.word,
        class_idx=request.class_idx,
        **{"class": request.class_name},
        morphs=request.morphs,
        meaning=request.meaning,
        example=request.example,
        source=request.source,
    ).where((keyword_table.c.id == request.id))

    await database.execute(query)
    return {"message": f"id: {request.id}가 word: {request.word}로 수정되었습니다"}


# 삭제
@router.delete(
    "/{id}",
    summary="핵심어 사전 삭제",
)
async def delete_keyword_dict(id: int = Path(description="ID", example=1)):
    """
    핵심어 사전 삭제
    """

    query = keyword_table.delete().where((keyword_table.c.id == id))

    await database.execute(query)
    return {"message": f"id: {id}가 삭제되었습니다"}
