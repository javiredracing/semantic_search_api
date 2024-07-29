from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field, model_validator, ConfigDict
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
from haystack.components.builders.prompt_builder import PromptBuilder

#from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
#from haystack.utils import print_documents
#from haystack.utils import print_answers
import os, io, copy
from datetime import datetime
from lib.ProcessText import *
import json
import time
import validators

#LLM_MODEL = "mixtral:8x7b-instruct-v0.1-q5_K_M"
LLM_MODEL = "llama3.1:8b-instruct-fp16"
#LLM_MODEL = "phi3:14b-medium-128k-instruct-f16"

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
        "name": "agents",
        "description": "Helpful tasks",
    },
]

#https://github.com/deepset-ai/haystack/tree/main/rest_api
#TRANSFORMER = "intfloat/multilingual-e5-base"
#TRANSFORMER = "mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es"
TRANSFORMER = 'intfloat/multilingual-e5-large-instruct'
#model = SentenceTransformer(TRANSFORMER,device="cuda")
#print(model)
EMBEDDINGS_SERVER = "http://0.0.0.0:7997/"
AUDIO_TRANSCRIBE_SERVER = "http://0.0.0.0:8080/"
DATABASE_SERVER = "http://localhost:9200"
OLLAMA_SERVER = "http://localhost:11434/"
AUDIO_PATH = "/home/administrador/audio2"


description = """Ask questions about the documents provided. 🚀
Model Max Sequence Length: **512**.
Services required: 
* infinity service (https://github.com/michaelfeil/infinity)
* Elastic search service.
* Ollama service.
* Diarization service."""

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
document_store = ElasticsearchDocumentStore(hosts = DATABASE_SERVER, index="semantic_search")

#document_embedder = SentenceTransformersDocumentEmbedder(model=TRANSFORMER,  normalize_embeddings=True )  
#document_embedder.warm_up()
#text_embedder = SentenceTransformersTextEmbedder(model=TRANSFORMER,  normalize_embeddings=True )
#text_embedder.warm_up()
bm25retriever = ElasticsearchBM25Retriever(document_store=document_store, scale_score=True)
retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)

#reader = ExtractiveReader(model="MMG/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es-finetuned-sqac",no_answer=False)
#reader.warm_up()
#PlanTL-GOB-ES/roberta-large-bne-sqac
  
#PrimitiveType = Union[str, int, float, bool]
class FilterRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    #filters: Optional[Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]] = None
    filters: Optional[Dict] = None
    
class InputParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    metadata: Optional[List[dict,]]= None   
        
    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
        
class uploadURL(InputParams):
    urls:List[str] = Field(title="URI")

class plainSRTParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text:str = Field(title="SRT text")
    filename:str = Field(title="File name", max_length=150)
    metadata:Optional[Dict] = None
        
class SearchQueryParam(FilterRequest):
    query:str = Field(title="Query max length 1500 chars", max_length=1500)
    top_k:int = Field(default=5,gt=0, le=50, description="Number of search results")
    context_size:int = Field(default=0,ge=0, le=20, description="Number of paragrhaps in context")

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
    if len(files) > 0:
        try:
            req = requests.post(AUDIO_TRANSCRIBE_SERVER + "transcribe/", json={"audio_path": files})  #transcribe audio files
            req.raise_for_status()
            #print(req.json())
        except requests.exceptions.RequestException as e:
            print("Error!", str(e))
            #return "Error!"

def processFile(file, ext, metadata):
    docs = []
    texts, pages = ProcessText.readFile(file.file.read(),ext)
    if ext == ProcessText.SRT_FORMAT:
        generateSRT_doc(file.filename, texts[0], metadata)
    else:
        generate_doc(texts, metadata, pages)
        
def generate_doc(texts, metadata, pages):
    docs = []
    if len(texts) > 0:
        try:
            req = requests.post(EMBEDDINGS_SERVER+"embeddings/", json={"input": texts, "model": TRANSFORMER})  #encode texts infinity api
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
                    
                    docs.append(Document(content=text.replace("\"", "\'"), meta=currentMetadata, embedding=doc_emb[i]["embedding"]))
        except requests.exceptions.RequestException as e:
            print("ERROR!")
                #TODO set logging error
    if len(docs) > 0:
        document_store.write_documents(docs)

#plain text srt to dict        
def parseSRT(texts):
    splitted_srt = texts.strip().split("\n\n")
    splitted_srt = [x for x in splitted_srt if x.strip()] #remove whites
    current_texts = []
    #split srt file
    for i, chunk in enumerate(splitted_srt, start=1):
        splitted_chunk = chunk.split("\n")
        if len(splitted_chunk) > 2:
            time = splitted_chunk[1].split("-->")
            start_time = time[0].strip()
            end_time = time[1].strip()
            content = ""
            for text_chunk in splitted_chunk[2:]:
                content += text_chunk + " "
            index_char = content.find(":")
            speaker = ""
            if index_char > 0 and index_char < 30 and len(content) > 0 and len(content) > (index_char + 1):
                speaker = content[0:index_char].strip()
                content = content[(index_char + 1):]
            current_texts.append({"content":content.strip(), "metadata":{"paragraph":i, "start_time":start_time, "end_time":end_time, "speaker":speaker}})

    return current_texts

def generateSRT_doc(filename, srt_text, metadata):
    #print(filename,metadata)
    srt_json = parseSRT(srt_text)
    
    docs = []
    
    current_texts = []
    for audio_chunk in srt_json:
        current_texts.append(audio_chunk["content"].replace("\"", "\'"))
        
    if len(current_texts) > 0:
        try:
            req = requests.post(EMBEDDINGS_SERVER+"embeddings", json={"input": current_texts, "model": TRANSFORMER})  #encode texts infinity api
            if req.status_code == 200:
                doc_emb = req.json()["data"]
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                for i, text in enumerate(current_texts):
                    currentMetadata = srt_json[i]["metadata"]
                    
                    currentMetadata.update(metadata)
                    currentMetadata.update({"timestamp":dt_string, "page":1, "name":filename, "file_type":"audio/"})
                    docs.append(Document(content=text, meta=currentMetadata, embedding=doc_emb[i]["embedding"]))
        except requests.exceptions.RequestException as e:
            print("ERROR!")
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
        try:
            req = requests.post(EMBEDDINGS_SERVER+"embeddings", json={"input": myquery,"model": TRANSFORMER})    #encode query           
            if req.status_code == 200:            
                query_embedding = req.json()["data"][0]["embedding"]
                #prediction = document_store.embedding_retrieval(query_embedding=query_embedding, top_k=params.top_k, filters=params.filters, return_embedding=False)
                prediction = retriever.run(query_embedding=query_embedding, top_k=params.top_k, filters=params.filters)["documents"]
                for doc in prediction:
                   del doc.embedding
        except requests.exceptions.RequestException as e:
            return None
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
            getDocuments = document_store.filter_documents(filters=current_filters)
            doc["before_context"] = []
            doc["after_context"] = []
            for context in getDocuments:
                if context.meta["paragraph"] < currentParagraph:
                    doc["before_context"].append(context.content)
                else:
                    doc["after_context"].append(context.content)
                    
     return docs

def launchAgent(prompt, options):
    try:
        #req = requests.post(OLLAMA_SERVER+"api/generate", json={"model":LLM_MODEL,"prompt":prompt, "system":"Eres un asistente muy obediente.","keep_alive":-1, "stream":False, "options": options})
        req = requests.post(OLLAMA_SERVER+"api/generate", json={"model":LLM_MODEL,"prompt":prompt, "keep_alive":-1, "stream":False, "options": options})
        req.raise_for_status()     
        return req.json()["response"]

    except requests.exceptions.RequestException as e:
        return None
    
    return None
        
def download_file(url):
    if validators.url(url):
        filename = url.split("/")[-1]
        audio_file = os.path.join(AUDIO_PATH, filename)
        if not os.path.isfile(audio_file):
            try:
                with requests.get(url, stream=True, timeout=10) as r:
                    with open(audio_file, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                return audio_file
            except requests.exceptions.RequestException as e:
                print("Error uploading the file " + filename + ": " + str(e))
            except Exception as err:
                print("Error uploading the file " + filename + ": " + str(err))
    return ""

@app.get("/", include_in_schema=False)
def main():
    return RedirectResponse(url='/docs')    

@app.get("/status/")
def check_status():
    #doc = document_store.describe_documents()
    result = {}
    try:
        res = requests.get(EMBEDDINGS_SERVER + "models/")
        res.raise_for_status()        
        result.update({"embeddings_status":res.json()})
    except requests.exceptions.RequestException as e:
        result.update({"embeddings_status":"ERROR! "+ str(e)})    
   
    try:
        res = requests.get(DATABASE_SERVER+"/_cluster/health")
        res.raise_for_status()
        result.update({"db_status":res.json()})
    except requests.exceptions.RequestException as e:  
        result.update({"db_status":"ERROR! "} + str(e))
    
    try:    
        res = requests.get(OLLAMA_SERVER+"api/ps")
        res.raise_for_status()
        result.update({"llm_status":res.json()})
    except requests.exceptions.RequestException as e:  
        result.update({"llm_status":"ERROR! " + str(e)})
    
    try:   
        res = requests.get(AUDIO_TRANSCRIBE_SERVER+"status/")
        res.raise_for_status()
        result.update({"diarization_status":res.json()})
    except requests.exceptions.RequestException as e:  
        result.update({"diarization_status":"ERROR! " +str(e)})

    return result


@app.get("/status/{audio_file}",tags=["audio"])
async def get_audio_status(audio_file: str) -> dict:

    '''Show status of a file audio in the transcription batch processing.
    * audio_file is the current audio that may be in process
    '''
    result = {
        "audio_file": audio_file,
        "status": "ERROR!"
    }
    try:   
        res = requests.get(AUDIO_TRANSCRIBE_SERVER+"status/" + audio_file)
        res.raise_for_status()
        result = res.json()
    except requests.exceptions.RequestException as e:
        result.update({"status":"ERROR! " +str(e)})
        
    return result


@app.post("/search/", tags=["search"])
def search_document(params:Annotated[SearchQueryParam, Body(embed=True)]):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the system to get a set of most relevant document pieces.
    Depending on the context size, it return previous and later paragraphs from documents
    
    Filter example:
    
    To get all documents you should provide an empty dict, like:`'{"filters": {}}`
    
    `{"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "meta.name", "operator": "==", "value": "2019"},
        			{"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]}
      					]
    			   }
  		      }`
    """
    docs = searchInDocstore(params)
    
    result = []
    if docs is None:
        return result

    for doc in docs:
        result.append(doc.to_dict())   
    docs_context = getContext(result, params.context_size)
    
    return docs_context


@app.post("/ask/", tags=["search"])
def ask_document(params:Annotated[SearchQueryParam, Body(embed=True)]):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the system to get a set of most releveant anwsers.
    
    Filter example:
    
    To get all documents you should provide an empty dict, like:`'{"filters": {}}`
    
    `{"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "meta.name", "operator": "==", "value": "2019"},
        			{"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]}
      					]
    			   }
  		      }`
    """
   
    myDocs = searchInDocstore(params)
    if myDocs is None:
        return {"answer":"None", "document":[]}        
    for doc in myDocs:
        doc.content = doc.content.replace('\r\n', '')
    
    template = """
    Con la información proporcionada, responde a la pregunta en el mismo lenguaje del texto proporcionado.
    Ignora tu conocimiento.

    Contexto:
    
    {% for document in myDocs %}
        {{ document.content }}
    {% endfor %}

    Pregunta: {{ query }}
    """
    builder = PromptBuilder(template=template)
    my_prompt = builder.run(myDocs=myDocs, query=params.query.strip())["prompt"].strip()
    options ={"num_predict": -1,"temperature": 0.1,"num_ctx":8192}    
    response=launchAgent(my_prompt, options)
    if response is None:
        response = ""
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
    
    `{"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "meta.name", "operator": "==", "value": "2019"},
        			{"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]}
      					]
    			   }
  		      }`
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
        
        `{"filters": { "operator": "AND",
                    "conditions": [
                        {"field": "meta.name", "operator": "==", "value": "2019"},
                        {"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]}
                            ]
                       }
                  }`
    """
     #print(filters.filters)
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
    
    `{"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "meta.name", "operator": "==", "value": "2019"},
        			{"field": "meta.companies", "operator": "in", "value": ["BMW", "Mercedes"]}
      					]
    			   }
  		      }`
    """
    prediction = document_store.filter_documents(filters=filters.filters)
    ids = [doc.id for doc in prediction]
    document_store.delete_documents(document_ids=ids)
    return True


@app.get("/agents/summarize/{file_name}", tags=["agents"], response_class=PlainTextResponse)  
def get_summary(file_name: str):
    """
    Devuelve un resumen que captura los puntos más importantes.
    """
    filters = {"field": "meta.name", "operator": "==", "value": file_name.strip()}
    prediction = document_store.filter_documents(filters=filters)
    
    template= '''
    Haz un resumen conciso y claro que capture los puntos más importantes del siguiente contexto. No te inventes nada. Responde solo con el resumen. Si no puedes hacerlo, no pongas nada.
    Contexto:
    {{myDocs}}
    
    Resumen:
    '''
    
    builder = PromptBuilder(template=template)
   
    options ={"num_predict": -1,"temperature": 0.1,"num_ctx":8192}
    
    if len(prediction) > 0:
        orderedlist = sorted(prediction, key=lambda doc: doc.meta['paragraph']) #order by paragraph
        custom_docs = ""
        current_speaker = ""
        response = []
        for doc in orderedlist:               
            if doc.meta["file_type"].startswith("audio/"): 
                if doc.meta['speaker'] != current_speaker:
                    if current_speaker != "" and not custom_docs.strip().endswith((".",",","!","?",";")):
                        custom_docs = custom_docs.strip() + "..."
                    custom_docs += f"\n\n- {doc.meta['speaker']}: " 
                    current_speaker = doc.meta['speaker']
            else:
                if not custom_docs.strip().endswith((".",",","!","?",";")) and doc.meta['paragraph'] != 0:
                    custom_docs = custom_docs.strip() + "..."
                custom_docs+= "\n\n"

            custom_docs += f"{doc.content} "
            
            if len(custom_docs) > 3500:
                if  doc.meta["file_type"].startswith("audio/"):
                    current_speaker = ""
                my_prompt = builder.run(myDocs=custom_docs)["prompt"].strip()
                #print(my_prompt)
                custom_docs = ""
                agent_response = launchAgent(my_prompt, options)
                if agent_response is None:
                    return ""
                response.append("\n\n" + agent_response)
                 
                
        if not custom_docs.strip().endswith((".",",","!","?",";")):
            custom_docs = custom_docs.strip() + "."
        if len(custom_docs) > 0:
            my_prompt = builder.run(myDocs=custom_docs)["prompt"].strip()
            custom_docs = ""
            agent_response = launchAgent(my_prompt, options)
            if agent_response is None:
                return ""
            response.append("\n\n" + agent_response)

        return  ''.join(response)
    else:
        raise HTTPException(status_code=404, detail="File not found!")


@app.get("/audio/get_SRT/{file_name}", tags=["audio"], response_class=PlainTextResponse)  
def get_SRT(file_name: str):
    """
    This endpoint get the srt file referred to a specific audio file that  have been processed previously.
    """
    filters = {"field": "meta.name", "operator": "==", "value": file_name.strip()}
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


@app.get("/audio/get_SRT_traslated/{file_name}/{lang}", tags=["audio"], response_class=PlainTextResponse) 
def get_SRT_traslated(file_name: str, lang:str):
    """
    This endpoint get the srt file referred to a specific audio file that have been processed previously, traslated into a defined language. The language param admit most commons languages.
    Ej: *Español, inglés, alemán, italiano, francés,...*
    """
    filters = {"field": "meta.name", "operator": "==", "value": file_name.strip()}
    prediction = document_store.filter_documents(filters=filters)
    if len(prediction) > 0:
        orderedlist = sorted(prediction, key=lambda d: d.meta['paragraph']) #order by paragraph
        #print("total docs",len(orderedlist))
        template="""
La siguiente lista contiene textos:

{{srt_file}}

Responde con una lista con cada texto traducido al idioma {{lang}}, en el mismo orden y con igual formato que la lista original. No te inventes nada.

Ejemplo respuesta:

1- Hello world.

2- A long sentence to translate.

"""

        builder = PromptBuilder(template=template)
        options ={"num_predict": -1,"temperature":0.1,"num_ctx":8192,"top_p":0.5}
        
        prompt_list = []
        srt_file = ""
        count = 0
        for i, doc in enumerate(orderedlist):
            if doc.meta["file_type"].startswith("audio/"):
                my_doc_mod= doc.content
                srt_file += f"{str(i)}- {my_doc_mod.capitalize()}\n\n"
                count += len(doc.content)
                if count >= 1500: #6000 chars is aprox 1500 tokens
                    my_prompt = builder.run(lang=lang, srt_file=srt_file)["prompt"].strip()
                    prompt_list.append(my_prompt)
                    srt_file = ""
                    count = 0
        
        if len(srt_file) > 0:
            my_prompt = builder.run(lang=lang, srt_file=srt_file)["prompt"].strip()
            prompt_list.append(my_prompt)            
          
        srt_file = []
        for prompt in prompt_list:
            agent_result = launchAgent(prompt, options)
            if agent_result is None:
                return ""
            data = agent_result.split("\n")
            #print(data)
            final_list = [i for i in data if i]
            for item in final_list[1:]:
                x = item.find("-")
                if x >= 1 and (x + 1) < len(item):
                    srt_file.append(item[(x+1):].strip())

#        print("total traslated paragraphs",len(srt_file))
        plain_res = ""
        for i, doc in enumerate(orderedlist):
            if doc.meta["file_type"].startswith("audio/"):
                plain_res += f"{str(i+1)}\n"
                text1=f"{doc.meta['start_time']} --> {doc.meta['end_time']}\n"
                plain_res += text1
                text1 = f"{doc.meta['speaker']}: {doc.content}\n"
                plain_res += text1
                if i < len(srt_file):
                    text1 = f"{doc.meta['speaker']}: {srt_file[i]}\n\n"
                else:
                    text1 = f"{doc.meta['speaker']}: --\n\n"
                plain_res += text1

                
        return plain_res
    else:
        raise HTTPException(status_code=404, detail="File not found!")


@app.post("/documents/upload/plainSRT/", tags=["docs"])
def upload_plain_srt(input_values:plainSRTParams, background_tasks: BackgroundTasks):
    """
    This endpoint accepts .srt files in plain text for processing.
    File name required.
    Metadata example for each file added: `{"user": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    background_tasks.add_task(generateSRT_doc, input_values.filename, input_values.text, input_values.metadata)
    return {"message":"SRT plain text uploaded correctly, processing..." , "filenames":input_values.filename}


@app.post("/audio/upload/", tags=["audio"])
def upload_audios(files: Annotated[List[UploadFile], File(description="files")], background_tasks: BackgroundTasks, metadata:InputParams = Body(...)):
    """
    This endpoint accepts audio in **.mp3** format. It only can be parsed by "\\n" character.
    
    `TODO: Add support for other parsing options.`
    
    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    #
    #https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    filenames =[]
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
    #print(filenames)
    background_tasks.add_task(audio_manager, filenames, metadata.metadata)

    return {"message":"Files uploaded correctly, processing..." , "filenames": [os.path.basename(file) for file in filenames]}


@app.post("/documents/upload/", tags=["docs"])
def upload_documents(files: Annotated[List[UploadFile], File(description="files")], background_tasks: BackgroundTasks, metadata:InputParams = Body(...)):
    """
    This endpoint accepts documents in **.doc, .pdf .xps, .epub, .mobi, .fb2, .cbz, .svg,.txt and.srt** format. It only can be parsed by paragraphs.
    
    `TODO: Add support for other parsing options.`
    
    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    #
    #https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    for file in files:
        if file.content_type.startswith("audio/"):
            return {"message":"Error uploading files! This endpint only accepts text files. Use /audio/upload/ endpoint for uploading audio files"}
    background_tasks.add_task(document_manager, copy.deepcopy(files), metadata.metadata)

    return {"message":"Files uploaded correctly, processing..." , "filenames": [file.filename for file in files]}

@app.post("/audio/upload/from_url", tags=["audio"])
def upload_audio_url(files:uploadURL, background_tasks:BackgroundTasks):
    filenames = []
    if len(files.urls) > 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for url in files.urls:
                futures.append(executor.submit(download_file, url))
            for future in concurrent.futures.as_completed(futures):
                filenames.append(future.result())
        filenames = [x for x in filenames if x] #clear emptys
        if len(filenames) > 0:
            background_tasks.add_task(audio_manager, filenames, files.metadata)    
            return {"message":"Files uploaded correctly, processing..." , "filenames": [os.path.basename(file) for file in filenames]}
    return {"message":"No files processed!"}

# @app.post("/documents/upload/from_url", tags=["docs"])
# def upload_documents_url(files:uploadURL, background_tasks:BackgroundTasks):
    
    # for url in files.urls:
        # if validators.url(url):
            # try:
                # res = requests.get(url)
                # res.raise_for_status()
            # except requests.exceptions.RequestException as e:
                # print("Error")
            