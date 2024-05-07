from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import PlainTextResponse
#from typing_extensions import Annotated
from pydantic import BaseModel, Extra, Field, model_validator
#from sentence_transformers import SentenceTransformer
import requests
import concurrent
import subprocess
import shutil
from typing import Dict, List, Optional, Union, Literal, Annotated

from haystack import Document
#from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever

#from haystack.components.readers import ExtractiveReader
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders.prompt_builder import PromptBuilder

#from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
#from haystack.utils import print_documents
#from haystack.utils import print_answers
import os, io, copy
from datetime import datetime
from lib.ProcessText import *
import json
import time

ROOT = '/home/administrador/audio'
tags_metadata = [
  #  {
  #      "name": "Question/answer docs",
  #      "description": "Get info from your documents",
  #  },
    {
        "name": "search",
        "description": "Get responses about docs",
        #"externalDocs": {
        #    "description": "Made with FastApi",
        #    "url": "https://fastapi.tiangolo.com/",
        #},
    },
    {
        "name": "docs",
        "description": "Manage documents",
    },
    {
        "name": "audio",
        "description": "Manage audio",
    },
    {
        "name": "utils",
        "description": "Helpful utilities",
    },
]

#https://github.com/deepset-ai/haystack/tree/main/rest_api
#TRANSFORMER = "intfloat/multilingual-e5-base"
#TRANSFORMER = "mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es"
TRANSFORMER = 'intfloat/multilingual-e5-large-instruct'
#model = SentenceTransformer(TRANSFORMER,device="cuda")
#print(model)
SERVER_URL=""
AUDIO_PATH="/home/administrador/audio"

description = """
Ask questions about the documents provided. 🚀
Model Max Sequence Length: **512**.
Required infinity service (https://github.com/michaelfeil/infinity)
"""

app = FastAPI(
    title="Semantic search service",
    description=description,
    summary="Semantic searchs in documents",
    version="2.0.0",
    #terms_of_service="http://example.com/terms/",
    contact={
        "name": "Javier Fernández",
        #"url": "http://iter.es/contact/",
        "email": "jfernandez@iter.es",
    },
    license_info={
        "name": "Apache 2.0",
        "identifier": "MIT",
    },openapi_tags=tags_metadata
    )

#document_store = InMemoryDocumentStore(embedding_similarity_function="cosine", bm25_algorithm="BM25Plus")
document_store = ElasticsearchDocumentStore(hosts = "http://localhost:9200", index="semantic_search")

#document_embedder = SentenceTransformersDocumentEmbedder(model=TRANSFORMER,  normalize_embeddings=True )  
#document_embedder.warm_up()
#text_embedder = SentenceTransformersTextEmbedder(model=TRANSFORMER,  normalize_embeddings=True )
#text_embedder.warm_up()
bm25retriever = ElasticsearchBM25Retriever(document_store=document_store, scale_score=True)
retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)

#reader = ExtractiveReader(model="MMG/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es-finetuned-sqac",no_answer=False)
#reader.warm_up()
#PlanTL-GOB-ES/roberta-large-bne-sqac
  
class RequestBaseModel(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid
        
PrimitiveType = Union[str, int, float, bool]
class FilterRequest(RequestBaseModel):
    filters: Optional[Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]] = None
    
class InputParams(RequestBaseModel):
    metadata: Optional[List[dict,]]= None   
        
    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
        
class SearchQueryParam(FilterRequest):
    query:str = Field(title="Query max length 1500 chars", max_length=1500)
    top_k:int = Field(default=5,gt=0, le=50, description="Number of search results")
    context_size:int = Field(default=0,ge=0, le=20, description="Number of paragrhaps in context")
    
class AskQueryParams(SearchQueryParam):
    top_k_answers:int = Field(default=2,gt=0, le=50, description="Number of ask results")

class AskFileName(RequestBaseModel):
    file_name:str = Field(title="File name", max_length=1500, min_length=3)

def document_manager(files, metadata):
    start = time.process_time()
    with concurrent.futures.ThreadPoolExecutor() as executor: # optimally defined number of threads
        for i,file in enumerate(files):
            nameParsed = file.filename.split(".")            
            if len(nameParsed) > 1:                                        
                meta = None
                if metadata and len(metadata) > i:
                    meta = metadata[i]
                executor.submit(processFile, file, nameParsed[1],meta)
                #processFile(file,nameParsed[1], meta)
            else:
                print("error extension1")
    
    end = time.process_time()
    print("Processing time:",end - start)

def audio_manager(files, metadata):
    
    start = time.process_time()
    #1. launch process to generate srt, txt and json outputs in audio path
    for i,file in enumerate(files):
        p = subprocess.run(["/home/administrador/whisper-diarization/venv/bin/python3", "/home/administrador/whisper-diarization/diarize_parallel.py", "-a", file[0], "--no-stem", "--whisper-model", "large-v3", "--device", "cuda", "--language", "es", "--batch-size", "32"])                        

    #2. read json outputs from audio path, encode and upload to database as documents
    with concurrent.futures.ThreadPoolExecutor() as executor: # optimally defined number of threads
        for i,file in enumerate(files):
            meta = {"file_type":file[1]}
            
            if metadata and len(metadata) > i:
                meta.update(metadata[i])
            executor.submit(processSRT_json, file[0], meta)
            #processSRT_json(file,meta)
    end = time.process_time()
    print("Processing time:",end - start)

def processFile(file, ext, metadata):
    docs = []
    texts, pages = ProcessText.readFile(file.file.read(),ext)               
    if len(texts) > 0:        
        req = requests.post("http://0.0.0.0:7997/embeddings", json={"input": texts, "model": TRANSFORMER})  #encode texts infinity api
        if req.status_code == 200:
            doc_emb = req.json()["data"]
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            for i, text in enumerate(texts):
                currentMetadata = {"name": file.filename, "paragraph":i, "timestamp":dt_string, "file_type":file.content_type}
                if metadata:
                    currentMetadata.update(metadata)
                if len(pages) == len(texts):
                    currentMetadata.update({"page":pages[i]})
                elif len(pages) == 1:
                    currentMetadata.update({"page":pages[0]})
                
                docs.append(Document(content=text, meta=currentMetadata, embedding=doc_emb[i]["embedding"]))
                
    if len(docs) > 0:
        document_store.write_documents(docs)

def processSRT_json(file,metadata):
    print(file,metadata)
    docs = []
    filename = file.split(".")
    audio_file = os.path.join(ROOT, filename[0]+".json")
    with open(audio_file, encoding='utf-8-sig') as json_file:
        json_dict = json.load(json_file)

    current_texts = []
    for audio_chunk in json_dict:
        current_texts.append(audio_chunk["content"])
        
    if len(current_texts) > 0:
        req = requests.post("http://0.0.0.0:7997/embeddings", json={"input": current_texts, "model": TRANSFORMER})  #encode texts infinity api
        if req.status_code == 200:
            doc_emb = req.json()["data"]
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            for i, text in enumerate(current_texts):
                currentMetadata = json_dict[i]["meta"]
                
                currentMetadata.update(metadata)
                currentMetadata.update({"timestamp":dt_string})
                docs.append(Document(content=text, meta=currentMetadata, embedding=doc_emb[i]["embedding"]))
    if len(docs) > 0:
        document_store.write_documents(docs)

def get_detailed_instruct(query: str) -> str:
    task_description = 'Given a web search query, retrieve relevant passages that answer the query'
    return f'Instruct: {task_description}\nQuery: {query}'
    
def searchInDocstore(params):
    count = params.query.strip().count(" ")+ 1
    prediction = None
    if  count < 3: 
        #print("BM25 method")
        prediction = bm25retriever.run(query=params.query.strip(),top_k=params.top_k, filters=params.filters)["documents"]
        #prediction = document_store.bm25_retrieval(query=params.query.strip(), top_k=params.top_k, filters=params.filters, scale_score=True)
        for doc in prediction:
           del doc.embedding
    else:
        myquery=get_detailed_instruct(params.query.strip())
        #embedding_query = text_embedder.run(text=myquery)["embedding"]
        #embedding_query = model.encode(myquery, normalize_embeddings=True, show_progress_bar=True, device="cuda", batch_size=4)
        req = requests.post("http://0.0.0.0:7997/embeddings", json={"input": myquery,"model": TRANSFORMER})    #encode query   
    
        if req.status_code == 200:            
            query_embedding = req.json()["data"][0]["embedding"]
            #prediction = document_store.embedding_retrieval(query_embedding=query_embedding, top_k=params.top_k, filters=params.filters, return_embedding=False)
            prediction = retriever.run(query_embedding=query_embedding, top_k=params.top_k, filters=params.filters)["documents"]
            for doc in prediction:
               del doc.embedding
    #print_documents(prediction, max_text_len=query.max_text_len, print_name=True, print_meta=True)
    return prediction

def getContext(docs, context_size):
     if context_size > 0:
        for doc in docs:            
            currentParagraph = doc["paragraph"]
            min_paragrah = currentParagraph - context_size
            if min_paragrah < 0:
                min_paragrah = 0
            min_paragrah = list(range(min_paragrah,currentParagraph))
            max_paragraph = list(range(currentParagraph + 1, currentParagraph + context_size + 1))
            context_list = min_paragrah + max_paragraph
            current_filters = {"operator": "AND", 
            "conditions": [{"field": "meta.name", "operator": "==", "value": doc["name"]},{"field": "meta.paragraph", "operator": "in", "value": context_list},]}
            #getDocuments = document_store.filter_documents(filters={"name":doc["name"], "paragraph": context_list})
            getDocuments = document_store.filter_documents(filters=current_filters)
            doc["before_context"] = []
            doc["after_context"] = []
            for context in getDocuments:
                if context.meta["paragraph"] < currentParagraph:
                    doc["before_context"].append(context.content)
                else:
                    doc["after_context"].append(context.content)
                    
     return docs

@app.get("/")
def main():
    content = """
<body>
<form action="/documents/upload/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)    
    
@app.get("/status/")
def check_status():
    #doc = document_store.describe_documents()
    #print(doc)
    return True


@app.post("/search/", tags=["search"])
def search_document(params:Annotated[SearchQueryParam, Body(embed=True)]):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the system to get a set of most relevant document pieces.
    Depending on the context size, it return previous and later paragraphs from documents
    
    Filter example:
    
    To get all documents you should provide an empty dict, like:`'{"filters": {}}`
    
    {"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "meta.name", "operator": "==", "value": "2019"},
        			{"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]},
      					]
    			   }
  		      }
    """
    docs = searchInDocstore(params)

    result = []
    for doc in docs:
        result.append(doc.to_dict())   
    docs_context = getContext(result, params.context_size)
    
    return docs_context

@app.post("/ask/", tags=["search"])
def ask_document(params:Annotated[AskQueryParams, Body(embed=True)]):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the system to get a set of most releveant anwsers.
    
    Filter example:
    
    To get all documents you should provide an empty dict, like:`'{"filters": {}}`
    
    {"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "meta.name", "operator": "==", "value": "2019"},
        			{"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]},
      					]
    			   }
  		      }
    """
   
    myDocs = searchInDocstore(params)
    for doc in myDocs:
        doc.content = doc.content.replace('\r\n', '')
   
    template = """
    Con la información proporcionada, responde a la pregunta en el lenguaje del texto proporcionado.
    Ignora tu conocimiento.

    Contexto:
    {% for document in myDocs %}
        {{ document.content }}
    {% endfor %}

    Pregunta: {{ query }}
    """
    builder = PromptBuilder(template=template)
    my_prompt = builder.run(myDocs=myDocs, query=params.query.strip())["prompt"].strip()
    options ={"num_predict": -1,"temperature": 0.1}
    req = requests.post("http://localhost:11434/api/generate", json={"model":"llama3:8b-instruct-fp16","prompt":my_prompt, "keep_alive": -1, "stream":False, "options": options})  #OLLAMA api
    response="No hay respuesta"
    if req.status_code == 200:
        response = req.json()["response"]
        #print(response)
    #result = reader.run(
    #    query=params.query.strip(),
    #    documents=myDocs,
    #    top_k=params.top_k_answers
    # )["answers"]

    #for answer in result:
    #    doc = answer.document.to_dict()
    #    answer.document=getContext([doc], params.context_size)
    result = []
    for doc in myDocs:
        result.append(doc.to_dict())   
    docs_context = getContext(result, params.context_size)   
    final_result = {"answer":response, "document":docs_context}
    return final_result
   
@app.post("/documents/show/", tags=["docs"])
def show_documents(filters: FilterRequest):
    """
    This endpoint allows you to retrieve documents contained in your document store.
    You can filter the documents to retrieve by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Filter example:
    
    To get all documents you should provide an empty dict, like:`'{"filters": {}}`
    
    {"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "meta.name", "operator": "==", "value": "2019"},
        			{"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]},
      					]
    			   }
  		      }
    """
    prediction = document_store.filter_documents(filters=filters.filters)
    result = []
    for doc in prediction:
        doc_dict= doc.to_dict()        
        del doc_dict["embedding"]
        result.append(doc_dict)

    #print_documents(prediction, max_text_len=100, print_name=True, print_meta=True)
    return result
    
@app.post("/documents/name_list/", tags=["docs"])
def list_documents_name(filters: FilterRequest):

     """
        This endpoint allows you to retrieve the filename of all documents loaded.
        You can filter the documents to retrieve by metadata (like the document's name),
        or provide an empty JSON object to clear the document store.

        Filter example:
        
        To get all documents you should provide an empty dict, like:`'{"filters": {}}`
        
        {"filters": { "operator": "AND",
                    "conditions": [
                        {"field": "meta.name", "operator": "==", "value": "2019"},
                        {"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]},
                            ]
                       }
                  }
    """
     print(filters.filters)
     prediction = document_store.filter_documents(filters=filters.filters)
     
     res = []
     for doc in prediction:
        res.append(doc.meta["name"])
     res = list(set(res))
     return res
    
@app.post("/documents/delete/", tags=["docs"])
def delete_documents(filters: FilterRequest):
    """
    This endpoint allows you to delete documents contained in your document store.
    You can filter the documents to delete by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Filter example:
    
    To get all documents you should provide an empty dict, like:`'{"filters": {}}`
    
    {"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "meta.name", "operator": "==", "value": "2019"},
        			{"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]},
      					]
    			   }
  		      }
    """
    if not filters.filters:
        prediction = document_store.filter_documents(filters=filters.filters)
        ids = [doc.id for doc in prediction]
        document_store.delete_documents(document_ids=ids)
        return True
    else:
        raise HTTPException(status_code=404, detail="Empty file name!") 

@app.get("/documents/get_summary/", tags=["utils"], response_class=PlainTextResponse)  
def get_summary(file_name: str):
    filters = {"field": "meta.name", "operator": "==", "value": file_name}
    prediction = document_store.filter_documents(filters=filters)
    
    if len(prediction) > 0:
        orderedlist = sorted(prediction, key=lambda doc: doc.meta['paragraph']) #order by paragraph
        custom_docs = ""
        current_speaker = ""
        for doc in orderedlist:               
            if doc.meta["file_type"].startswith("audio/"): 
                if doc.meta['speaker'] != current_speaker:
                    if current_speaker != "" and not custom_docs.strip().endswith((".",",","!","?",";")):
                        custom_docs = custom_docs.strip() + "."
                    custom_docs += f"\n\n- {doc.meta['speaker']}: " 
                    current_speaker = doc.meta['speaker']
            else:
                if not custom_docs.strip().endswith((".",",","!","?",";")) and doc.meta['paragraph'] != 0:
                    custom_docs = custom_docs.strip() + "."
                custom_docs+= "\n\n"
            custom_docs += f"{doc.content} "
        
        if not custom_docs.strip().endswith((".",",","!","?",";")):
            custom_docs = custom_docs.strip() + "."
        

        template= '''
        Haz una nota de prensa del siguiente rueda de prensa donde intervienen varios speakers. No te inventes nada.
        
        Rueda de prensa:
        
        {{myDocs}}
        
        Nota de prensa:
        '''
        
        builder = PromptBuilder(template=template)
        #my_prompt = builder.run(myDocs=custom_docs)["prompt"].strip()
        my_prompt = builder.run(myDocs=custom_docs)["prompt"].strip()
        options ={"num_predict": -1,"temperature": 0.7}
        req = requests.post("http://localhost:11434/api/generate", json={"model":"llama3:8b-instruct-fp16","prompt":my_prompt, "keep_alive":-1, "stream":False, "options": options})
        if req.status_code == 200:
            return req.json()["response"]
        else:
            raise HTTPException(status_code=204, detail="Error generating summary") 
    else:
        raise HTTPException(status_code=404, detail="File not found!")

@app.get("/audio/get_SRT/", tags=["audio"], response_class=PlainTextResponse)  
def get_SRT(file_name: str):
    """
    This endpoint get the srt file referred to a specific audio file that  have been processed previously.
    """
    filters = {"field": "meta.name", "operator": "==", "value": file_name}
    prediction = document_store.filter_documents(filters=filters)
    if len(prediction) > 0:
        orderedlist = sorted(prediction, key=lambda d: d.meta['paragraph']) #order by paragraph
        srt_file = ""
        for i, doc in enumerate(orderedlist, start=1):
            if doc.meta["file_type"].startswith("audio/"):
                srt_file += f"{i}\n"
                text1=f"{doc.meta['start_time']} --> {doc.meta['end_time']}\n"
                srt_file += text1
                text1 = f"{doc.meta['speaker']}: {doc.content}\n\n"
                srt_file += text1
        return srt_file
    else:
        raise HTTPException(status_code=404, detail="File not found!") 

@app.post("/audio/upload/", tags=["audio"])
def upload_audios(files: Annotated[List[UploadFile], File(description="files")], background_tasks: BackgroundTasks, metadata:InputParams = Body(...)):
    """
    This endpoint accepts audio in .mp3 format. It only can be parsed by "\\n" character.
    
    `TODO: Add support for other parsing options.`
    
    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    #
    #https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    filenames =[]
    for file in files:
        if file.content_type.startswith("audio/"):
            audio_file = os.path.join(ROOT, file.filename)
            if os.path.isfile(audio_file):
                print('File exists')
            else:               
                with open(audio_file, "wb") as buffer:                
                    shutil.copyfileobj(file.file, buffer)
                    filenames.append((file.filename, file.content_type))    
                
    background_tasks.add_task(audio_manager, filenames, metadata.metadata)

        
    return {"message":"Files uploaded correctly, processing..." , "filenames": [file[0] for file in filenames]}

    
@app.post("/documents/upload/", tags=["docs"])
def upload_documents(files: Annotated[List[UploadFile], File(description="files")], background_tasks: BackgroundTasks, metadata:InputParams = Body(...)):
    """
    This endpoint accepts documents in .doc, .pdf .xps, .epub, .mobi, .fb2, .cbz, .svg and .txt format at this time. It only can be parsed by "\\n" character.
    
    `TODO: Add support for other parsing options.`
    
    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    #
    #https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    background_tasks.add_task(document_manager, copy.deepcopy(files), metadata.metadata)
    #

        
    return {"message":"Files uploaded correctly, processing..." , "filenames": [file.filename for file in files]}
