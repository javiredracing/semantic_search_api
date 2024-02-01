from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from pydantic import BaseModel, Extra, Field

from typing import Dict, List, Optional, Union, Literal

import pandas as pd
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader, BM25Retriever
from sentence_transformers import SentenceTransformer
#from haystack.utils import print_documents
#from haystack.utils import print_answers
import torch, os, io
from datetime import datetime
from lib.ProcessText import *
import json

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
]

PREFIX_PASSAGE = "passage: "
PREFIX_QUERY = "query: "
#https://github.com/deepset-ai/haystack/tree/main/rest_api
TRANSFORMER = "intfloat/multilingual-e5-small"
#TRANSFORMER = "mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es"
model = SentenceTransformer(TRANSFORMER)
print(model)

description = """
Ask questions about the documents provided. 🚀
## Configs:
Model Max Sequence Length: **{}**.
Torch version: **{}**.
Is CUDA enabled? **{}**.
""".format(model.max_seq_length, torch.__version__, torch.cuda.is_available())

app = FastAPI(
    title="Semantic search service",
    description=description,
    summary="Semantic searchs in documents",
    version="0.0.1",
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

document_store = InMemoryDocumentStore(use_bm25=True, similarity='cosine', embedding_dim=model[1].word_embedding_dimension)   #large_model=1024, small=384
retriever1 = EmbeddingRetriever(
    document_store=document_store,
    embedding_model=TRANSFORMER,
    use_gpu=True,
    scale_score=True,
)
retriever2 = BM25Retriever(document_store=document_store)
reader = FARMReader("MMG/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es-finetuned-sqac", use_gpu=True, context_window_size=3000) 
#PlanTL-GOB-ES/roberta-large-bne-sqac
  
class RequestBaseModel(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid
        
PrimitiveType = Union[str, int, float, bool]
class FilterRequest(RequestBaseModel):
    filters: Optional[Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]] = None
    
class InputParams(RequestBaseModel):
    metadata: List[Union[None, dict, ]] = None   
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
        
class SearchQueryParam(FilterRequest):
    query:str = Field(title="Query max length 1500 chars", max_length=1500)
    #max_text_len:int = Field(default=1000 ,gt=0, le=1500, description="Max length of results text")
    top_k:int = Field(default=5,gt=0, le=50, description="Number of search results")
    context_size:int = Field(default=0,ge=0, le=20, description="Number of paragrhaps in context")
    
class AskQueryParams(SearchQueryParam):
    top_k_answers:int = Field(default=2,gt=0, le=50, description="Number of ask results")

def document_manager(files, metadata, force_update):
    docs = []
    for i,file in enumerate(files):
        nameParsed = file.filename.split(".")
        if len(nameParsed) > 1:                           
            splitted_text, text_pages = ProcessText.readFile(file.file.read(),nameParsed[1])
            if len(splitted_text) == 0:
                return("error")            
            else:
                if force_update:
                    document_store.delete_documents(filters={'name': file.filename})
                meta = None
                if metadata and len(metadata) > i:
                    meta = metadata[i]
                docs = docs + generateEmbeddings(splitted_text, text_pages, file.filename, meta)
                
        else:
            print("error extension1")
            #break

    if len(docs) > 0:    
        document_store.write_documents(docs)
 
def generateEmbeddings(texts, pages, filename, metadata = None):
    docs = []
    if len(texts) > 0:
        embeddings = model.encode([PREFIX_PASSAGE + i.replace('\r\n', '') for i in texts], normalize_embeddings=True, show_progress_bar=True)     
        #end = time.process_time()
        #print("Processing time:",end - start)
        table_with_embeddings = pd.DataFrame(
            {"content": texts, "embedding": embeddings.tolist()}
        )        
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        # select from table and build in memory search pipeline without re-computing embeddings
        for i, row in table_with_embeddings.iterrows():
            currentMetadata = {'name': filename, "paragraph":i, "timestamp":dt_string}
            if metadata:
                currentMetadata.update(metadata)
            if len(pages) == len(texts):
                currentMetadata.update({"page":pages[i]})
            elif len(pages) == 1:
                currentMetadata.update({"page":pages[0]})
                
            docs.append(Document(content=row["content"], embedding=row["embedding"], content_type="text", meta=currentMetadata))
    return docs
    
def searchInDocstore(params):
    count = params.query.strip().count(" ")+ 1
    prediction = None
    if  count < 3: 
        #print("BM25 method")
        prediction = retriever2.retrieve(
                query=params.query.strip(),
                top_k=params.top_k,
                filters=params.filters
            )
        for doc in prediction:
            doc.embedding = None
    else:
        #print("model method")
        prediction = retriever1.retrieve(
            query= PREFIX_QUERY + params.query.strip(),
            top_k=params.top_k,
            filters=params.filters
        )
    #print_documents(prediction, max_text_len=query.max_text_len, print_name=True, print_meta=True)
    return prediction

def getContext(docs, context_size):
     if context_size > 0:
        for doc in docs:            
            currentParagraph = doc["meta"]["paragraph"]
            min_paragrah = currentParagraph - context_size
            if min_paragrah < 0:
                min_paragrah = 0
            min_paragrah = list(range(min_paragrah,currentParagraph))
            max_paragraph = list(range(currentParagraph + 1, currentParagraph + context_size + 1))
            context_list = min_paragrah + max_paragraph
            getDocuments = document_store.get_all_documents(filters={"name":doc["meta"]["name"], "paragraph": context_list}, return_embedding=False)
            #print(getDocuments)
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
    doc = document_store.describe_documents()
    #print(doc)
    return doc


@app.post("/search/", tags=["search"])
def search_document(params:Annotated[SearchQueryParam, Body(embed=True)]):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the system to get a set of most relevant document pieces.
    Depending on the context size, it return previous and later paragraphs from documents
    
    Filter example:
    
    To get all documents you should provide an empty dict, like:`'{"filters": {}}`
    
    **Simple** `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`
    
    **Advanced** `{"filters":{
    "type": "article",
    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
    "rating": {"$gte": 3},
    "$or": {
        "genre": ["economy", "politics"],
        "publisher": "nytimes"
    }
    }}`
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
    
    **Simple** `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`
    
    **Advanced** `{"filters":{
    "type": "article",
    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
    "rating": {"$gte": 3},
    "$or": {
        "genre": ["economy", "politics"],
        "publisher": "nytimes"
    }
    }}`
    """
   
    myDocs = searchInDocstore(params)
    for doc in myDocs:
        doc.content = doc.content.replace('\r\n', '')
        
    result = reader.predict(
        query=params.query,
        documents=myDocs,
        top_k=params.top_k_answers
    )
    #add meta to answers
    for answer in result['answers']:
        for id in answer.document_ids:
            for doc in myDocs:            
                if doc.id == id:
                    answer.meta.update(doc.meta)
                    break

    anwsers = []
    for doc in result['answers']:
        anwsers.append(doc.to_dict())    
    docs_context = getContext(anwsers, params.context_size)
    return docs_context    
   
@app.post("/documents/show/", tags=["docs"])
def show_documents(filters: FilterRequest, index: Optional[str] = None):
    """
    This endpoint allows you to retrieve documents contained in your document store.
    You can filter the documents to retrieve by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Filter example:
    
    To get all documents you should provide an empty dict, like:`'{"filters": {}}`
    
    **Simple** `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`
    
    **Advanced** `{"filters":{
    "type": "article",
    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
    "rating": {"$gte": 3},
    "$or": {
        "genre": ["economy", "politics"],
        "publisher": "nytimes"
    }
    }}`
    """
    prediction = document_store.get_all_documents(filters=filters.filters, index=index, return_embedding=False)
    #print(prediction)
    #print_documents(prediction, max_text_len=100, print_name=True, print_meta=True)
    return prediction
    
@app.post("/documents/name_list/", tags=["docs"])
def list_documents_name(filters: FilterRequest, index: Optional[str] = None):

     """
        This endpoint allows you to retrieve the filename of all documents loaded.
        You can filter the documents to retrieve by metadata (like the document's name),
        or provide an empty JSON object to clear the document store.

        Filter example:
        
        To get all documents you should provide an empty dict, like:`'{"filters": {}}`
        
        **Simple** `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`
        
        **Advanced** `{"filters":{
        "type": "article",
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
        "$or": {
            "genre": ["economy", "politics"],
            "publisher": "nytimes"
        }
        }}`
    """
     prediction = document_store.get_all_documents(filters=filters.filters, index=index, return_embedding=False)
     res = []
     for doc in prediction:
        res.append(doc.meta["name"])
     res = list(set(res))
     return res
    
@app.post("/documents/delete/", tags=["docs"])
def delete_documents(filters: FilterRequest, index: Optional[str] = None):
    """
    This endpoint allows you to delete documents contained in your document store.
    You can filter the documents to delete by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Filter example:
    
    To get all documents you should provide an empty dict, like:`'{"filters": {}}`
    
    **Simple** `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`
    
    **Advanced** `{"filters":{
    "type": "article",
    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
    "rating": {"$gte": 3},
    "$or": {
        "genre": ["economy", "politics"],
        "publisher": "nytimes"
    }
    }}`
    """
    document_store.delete_documents(filters=filters.filters, index=index)
    return True

@app.post("/documents/upload/", tags=["docs"])
def upload_documents(files: Annotated[List[UploadFile], File(description="files")], background_tasks: BackgroundTasks, metadata:InputParams = Body(None) , force_update:Annotated[bool, Form(description="Remove older files")] = True):
    """
    This endpoint accepts documents in .pdf .xps, .epub, .mobi, .fb2, .cbz, .svg and .txt format at this time. It only can be parsed by "\\n" character.
    
    `TODO: Add support for other document formats (DOC,...) and more parsing options.`
    
    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    #
    #https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    background_tasks.add_task(document_manager, files, metadata.metadata, force_update)

    return {"message":"Files uploaded correctly, processing..." , "filenames": [file.filename for file in files]}