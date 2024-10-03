from __future__ import annotations

import concurrent
import copy
import json
import os
import shutil
from typing import Optional, Dict, Annotated, List

from fastapi import APIRouter, Body, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.apis.api_documents.write_docs import generateSRT_doc, document_manager, audio_manager, download_file
from app.core.auth import decode_access_token
from app.core.config import AUDIO_PATH


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

class uploadURL(InputParams):
    urls:List[str] = Field(title="URI")

router = APIRouter()

class plainSRTParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text:str = Field(title="SRT text")
    filename:str = Field(title="File name", max_length=150)
    metadata:Optional[Dict] = None
    token:str = Field(title="Access token", max_length=175)

@router.post("/documents/upload/plainSRT/", tags=["documents"])
def upload_plain_srt(input_values:plainSRTParams, background_tasks: BackgroundTasks) -> dict:
    """
    Accepts .srt files in plain text for processing.
    File name required.
    Metadata example for each file added: `{"user": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """

    index = decode_access_token(input_values.token)
    print(index)

    background_tasks.add_task(generateSRT_doc, input_values.filename, input_values.text, input_values.metadata, index)
    return {"message":"SRT plain text uploaded correctly, processing..." , "filenames":input_values.filename}


@router.post("/documents/upload/", tags=["documents"])
def upload_documents(files: Annotated[List[UploadFile], File(description="files")], background_tasks: BackgroundTasks,
                     metadata: InputParams = Body(...)):
    """
    Accepts documents in **.doc, .pdf .xps, .epub, .mobi, .fb2, .cbz, .svg,.txt and .srt** format. It only can be parsed by paragraphs.

    `TODO: Add support for other parsing options.`

    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    index = decode_access_token(metadata.token)
    print(index)
    #
    # https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    for file in files:
        if file.content_type.startswith("audio/"):
            return {
                "message": "Error uploading files! This endpint only accepts text files. Use /audio/upload/ endpoint for uploading audio files"}
    background_tasks.add_task(document_manager, copy.deepcopy(files), metadata.metadata, index)


@router.post("/audio/upload/", tags=["audio"])
def upload_audios(files: Annotated[List[UploadFile], File(description="files")], background_tasks: BackgroundTasks,
                  metadata: InputParams = Body(...)):
    """
    This endpoint accepts audio in **.mp3** format. It only can be parsed by "\\n" character.

    `TODO: Add support for other parsing options.`

    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    index = decode_access_token(metadata.token)
    print(index)
    #
    # https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    filenames = []
    for file in files:
        if file.content_type.startswith("audio/"):
            audio_file = os.path.join(AUDIO_PATH, file.filename)
            if os.path.isfile(audio_file):
                print('File exists')
            else:
                try:
                    with open(audio_file, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                        filenames.append(audio_file)
                except Exception:
                    print("There was an error uploading the file")
                finally:
                    file.file.close()
    # print(filenames)
    background_tasks.add_task(audio_manager, filenames, metadata.metadata, index)

    return {"message": "Files uploaded correctly, processing...",
            "filenames": [os.path.basename(file) for file in filenames]}


@router.post("/audio/upload/from_url", tags=["audio"])
def upload_audio_url(files: uploadURL, background_tasks: BackgroundTasks):
    """
    This endpoint accepts documents in **.doc, .pdf .xps, .epub, .mobi, .fb2, .cbz, .svg,.txt and.srt** format from external URLs It only can be parsed by paragraphs.

    `TODO: Add support for other parsing options.`

    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    index = decode_access_token(files.token)
    print(index)

    filenames = []
    if len(files.urls) > 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for url in files.urls:
                futures.append(executor.submit(download_file, url))
            for future in concurrent.futures.as_completed(futures):
                filenames.append(future.result())
        filenames = [x for x in filenames if x]  # clear emptys
        if len(filenames) > 0:
            background_tasks.add_task(audio_manager, filenames, files.metadata)
            return {"message": "Files uploaded correctly, processing...",
                    "filenames": [os.path.basename(file) for file in filenames]}
    return {"message": "No files processed!"}