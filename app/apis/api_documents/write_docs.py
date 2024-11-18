from __future__ import annotations

import os
import logging

from datetime import datetime

import requests
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from pydantic import FileUrl

from app.apis.api_documents.ProcessText import ProcessText
from app.core.config import EMBEDDINGS_SERVER, EMBEDDINGS_MODEL, DB_HOST

def download_process_file(url_file:FileUrl, metadata:dict, index:str):
    try:
        response = requests.get(url_file)
        if response.status_code == 200:
            filename = response.headers.get("Content-Disposition")
            if filename:
                filename = filename.split("=")[-1].strip().strip('"')
            else:
                filename = os.path.basename(url_file.path)
            content_type = response.headers.get("Content-Type")
            if len(filename.split(".")) > 0:
                if not content_type:
                    content_type = filename.split(".")[1]
                processFile(file=response.content,filename=filename, content_type=content_type, metadata=metadata, index=index)
    except requests.RequestException:
        pass
    return None

def processFile(file:bytes, filename:str, content_type:str, metadata:dict, index:str):
    nameParsed = filename.split(".")
    if len(nameParsed) > 1:
        texts, pages = ProcessText.readFile(file, nameParsed[1])
        if nameParsed[1] == ProcessText.SRT_FORMAT:
            docs = generateSRT_doc(filename=filename, srt_text= texts[0], metadata=metadata)
        else:
            docs = generate_doc(filename=filename, content_type=content_type, texts=texts, metadata=metadata, pages=pages)
        save_docs(docs=docs,index=index)

def processPlainText(filename:str, plain_text:str, metadata:dict, index:str):
    ext = filename.split(".")
    if len(ext) > 0:
        if ext[1] == ProcessText.SRT_FORMAT:
            docs = generateSRT_doc(filename= filename, srt_text=plain_text, metadata=metadata)
            save_docs(docs=docs,index=index)
        elif ext[1] == ProcessText.VALID_FORMATS:
            texts, pages = ProcessText.read_plain_text(plain_text=plain_text)
            docs = generate_doc(filename=filename, content_type="text/plain", texts=texts, metadata=metadata, pages=pages)
            save_docs(docs=docs,index=index)

def save_docs(docs:list, index:str):
    if len(docs) > 0:
        document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search")  # TODO: Replace index
        document_store.write_documents(documents=docs,policy=DuplicatePolicy.SKIP)
        document_store.client.close()

def parseSRT(texts:str) -> list:
    """
    Parse plain text srt to dict
    """
    splitted_srt = texts.strip().split("\n\n")
    splitted_srt = [x for x in splitted_srt if x.strip()] #remove whites
    current_texts = []
    #split srt file
    for i, chunk in enumerate(splitted_srt, start=1):
        splitted_chunk = chunk.split("\n")
        if len(splitted_chunk) > 2:
            full_time = splitted_chunk[1].split("-->")
            start_time = full_time[0].strip()
            end_time = full_time[1].strip()
            content = ""
            for text_chunk in splitted_chunk[2:]:
                content += text_chunk + " "
            index_char = content.find(":")
            speaker = ""
            if 0 < index_char < 30 and len(content) > 0 and len(content) > (index_char + 1):
                speaker = content[0:index_char].strip()
                content = content[(index_char + 1):]
            current_texts.append({"content":content.strip(), "metadata":{"paragraph":i, "start_time":start_time, "end_time":end_time, "speaker":speaker}})

    return current_texts


def generateSRT_doc(filename:str, srt_text:str, metadata:dict) -> list:
    # print(filename,metadata)
    srt_json = parseSRT(srt_text)
    docs = []

    current_texts = []
    for srt_chunk in srt_json:
        current_texts.append(srt_chunk["content"].replace("\"", "\'"))

    if len(current_texts) > 0:
        try:
            req = requests.post(EMBEDDINGS_SERVER + "embeddings",
                                json={"input": current_texts, "model": EMBEDDINGS_MODEL})  # encode texts infinity api
            if req.status_code == 200:
                doc_emb = req.json()["data"]
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                for i, text in enumerate(current_texts):
                    currentMetadata = srt_json[i]["metadata"]
                    currentMetadata.update(metadata)
                    currentMetadata.update({"timestamp": dt_string, "page": 1, "name": filename, "file_type": "srt"})
                    docs.append(Document(content=text, meta=currentMetadata, embedding=doc_emb[i]["embedding"]))
        except requests.exceptions.RequestException as e:
            logging.error(e.response.text)

    return docs

def generate_doc(filename:str, content_type:str, texts:list, metadata:dict, pages:list) -> list:
    docs = []
    if len(texts) > 0:
        try:
            req = requests.post(EMBEDDINGS_SERVER + "embeddings/",
                                json={"input": texts, "model": EMBEDDINGS_MODEL})  # encode texts infinity api
            if req.status_code == 200:
                doc_emb = req.json()["data"]
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                for i, text in enumerate(texts):
                    currentMetadata = {"name": filename, "paragraph": i, "timestamp": dt_string,
                                       "file_type": content_type}
                    if metadata:
                        currentMetadata.update(metadata)
                    if len(pages) == len(texts):
                        currentMetadata.update({"page": pages[i]})
                    elif len(pages) == 1:
                        currentMetadata.update({"page": pages[0]})

                    docs.append(Document(content=text.replace("\"", "\'"), meta=currentMetadata,
                                         embedding=doc_emb[i]["embedding"]))
        except requests.exceptions.RequestException as e:
            logging.error(e.response.text)

    return docs

