from __future__ import annotations

import copy
import json
from typing import Optional, Dict, Annotated, List

from fastapi import APIRouter, Body, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.apis.api_documents.write_docs import generateSRT_doc, document_manager
from app.core.auth import decode_access_token


class InputParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    token: str = Field(title="Access token", max_length=175)
    metadata: Optional[List[dict,]] = None

    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class PlainSRTParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text:str = Field(title="SRT text")
    filename:str = Field(title="File name", max_length=150)
    metadata:Optional[Dict] = None
    token:str = Field(title="Access token", max_length=175)

class ReturnUpload(BaseModel):
    message:str
    filenames:str

router = APIRouter()

@router.post("/upload/plainSRT/", tags=["documents"])
def upload_plain_srt(input_values:PlainSRTParams, background_tasks: BackgroundTasks) -> ReturnUpload:
    """
    Process .srt files in plain text.
    - text: Plain srt file
    - File name required.
    - Metadata example for each file added: `{"user": ["some", "more"], "category": ["only_one"]}`    OR    empty
    - Token access required
    """

    index = decode_access_token(input_values.token)
    print(index)

    background_tasks.add_task(generateSRT_doc, input_values.filename, input_values.text, input_values.metadata, index)
    return ReturnUpload(message="SRT plain text uploaded correctly, processing...", filenames=input_values.filename)


@router.post("/upload/", tags=["documents"])
def upload_documents(files: Annotated[List[UploadFile], File(description="Document files to upload")], background_tasks: BackgroundTasks,
                     metadata: InputParams = Body(...)) -> ReturnUpload:
    """
    Accepts documents in **.doc, .pdf .xps, .epub, .mobi, .fb2, .cbz, .svg, .txt and .srt** format. It only can be parsed by paragraphs.

    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    #TODO: Add support for other parsing options.
    index = decode_access_token(metadata.token)
    print(index)
    #
    # https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    for file in files:
        if file.content_type.startswith("audio/"):
            return ReturnUpload(message="Error uploading files! This endpoint only accepts text format files.", filenames=file.filename)

    background_tasks.add_task(document_manager, copy.deepcopy(files), metadata.metadata, index)
    return ReturnUpload(message="Processing documents", filenames="".join([file.filename for file in files]))

#TODO upload from url