from __future__ import annotations

import concurrent
import time
import logging

from datetime import datetime
import requests
from haystack import Document
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

from app.apis.api_documents.ProcessText import ProcessText
from app.core.config import EMBEDDINGS_SERVER, EMBEDDINGS_MODEL, DB_HOST



def document_manager(files, metadata, index:str):
    start = time.process_time()
    with concurrent.futures.ThreadPoolExecutor() as executor:  # optimally defined number of threads
        for i, file in enumerate(files):
            nameParsed = file.filename.split(".")
            if len(nameParsed) > 1:
                meta = None
                if metadata and len(metadata) > i:
                    meta = metadata[i]
                executor.submit(processFile, file, nameParsed[1], meta, index)
                # processFile(file,nameParsed[1], meta)
            else:
                logging.warning("Error! no valid extension")

    end = time.process_time()
    logging.info("Processing time:", end - start)


def processFile(file, ext, metadata, index:str):
    texts, pages = ProcessText.readFile(file.file.read(),ext)
    if ext == ProcessText.SRT_FORMAT:
        generateSRT_doc(file.filename, texts[0], metadata, index)
    else:
        generate_doc(file, texts, metadata, pages, index)


#plain text srt to dict
def parseSRT(texts:str) -> []:
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


def generateSRT_doc(filename:str, srt_text:str, metadata:dict, index:str):
    # print(filename,metadata)
    srt_json = parseSRT(srt_text)
    docs = []

    current_texts = []
    for audio_chunk in srt_json:
        current_texts.append(audio_chunk["content"].replace("\"", "\'"))

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
    if len(docs) > 0:
        document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search")  # TODO: Replace index
        document_store.write_documents(docs)
        document_store.client.close()


def generate_doc(file, texts, metadata, pages, index:str):
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
                    currentMetadata = {"name": file.filename, "paragraph": i, "timestamp": dt_string,
                                       "file_type": file.content_type}
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

    if len(docs) > 0:
        document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search")  # TODO: Replace index
        document_store.write_documents(docs)
        document_store.client.close()
