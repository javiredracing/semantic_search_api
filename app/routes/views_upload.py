from __future__ import annotations

import copy
import json
import os.path
from typing import Optional, Dict, Annotated, List

from fastapi import APIRouter, Body, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, ConfigDict, Field, model_validator, HttpUrl

from app.apis.api_documents.write_docs import download_process_file, processFile, processPlainText
from app.core.auth import decode_access_token


class InputParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    token: str = Field(title="Access token", max_length=175)
    metadata: Optional[dict] = None

    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class InputParamsURL(InputParams):
    url_files: list[HttpUrl]

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

@router.post("/upload/plainText/", tags=["documents"])
def upload_plain_text(input_values:PlainSRTParams, background_tasks: BackgroundTasks) -> ReturnUpload:
    """
    Process .srt and .txt files in plain text.
    - text: Plain text file
    - File name required.
    - Metadata example for each file added: `{"user": ["some", "more"], "category": ["only_one"]}`    OR    empty
    - Access token required
    """
    index = decode_access_token(input_values.token)
    print(index)
    background_tasks.add_task(processPlainText, input_values.filename, input_values.text, input_values.metadata, index)
    return ReturnUpload(message="Plain text uploaded correctly, processing...", filenames=input_values.filename)


@router.post("/upload/", tags=["documents"])
def upload_documents(files: Annotated[List[UploadFile], File(description="Document files to upload")], background_tasks: BackgroundTasks,
                     params: InputParams = Body(...)) -> ReturnUpload:
    """
    Accepts documents in **.doc, .pdf .xps, .epub, .mobi, .fb2, .cbz, .svg, .txt and .srt** format. It only can be parsed by paragraphs.
    - Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    - Access token required
    """
    #TODO: Add support for other parsing options.
    index = decode_access_token(params.token)
    print(index)
    filenames= []
    # https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    for file in files:
        if not file.content_type.startswith("audio/"):
            upload_file = copy.deepcopy(file)
            f = upload_file.file.read()
            background_tasks.add_task(processFile, f, file.filename, file.content_type,params.metadata, index)
            filenames.append(file.filename)
    return ReturnUpload(message=f"Processing {str(len(filenames))} documents", filenames="".join(filenames))


@router.post("/upload_url/", tags=["documents"])
def upload_documents_from_url(params: InputParamsURL, background_tasks: BackgroundTasks) -> ReturnUpload:
    """
    Accepts accessible URL files in **.doc, .pdf .xps, .epub, .mobi, .fb2, .cbz, .svg, .txt and .srt** format. It only can be parsed by paragraphs.
    - Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    - Access token required
    """
    index = decode_access_token(params.token)
    print(index)
    for url_file in params.url_files:
        background_tasks.add_task(download_process_file, url_file, params.metadata, index)
    return ReturnUpload(message=f"Processing {str(len(params.url_files))} documents", filenames="".join([os.path.basename(url.path) for url in params.url_files]))