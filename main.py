from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body
from fastapi.responses import HTMLResponse
#from typing_extensions import Annotated
from pydantic import BaseModel, Extra, Field, model_validator
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Union, Literal, Annotated

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.readers import ExtractiveReader
#from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
#from haystack.utils import print_documents
#from haystack.utils import print_answers
import torch, os, io, copy
from datetime import datetime
from lib.ProcessText import *
import json
import time

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

#PREFIX_PASSAGE = "passage: "
#PREFIX_QUERY = "query: "
#https://github.com/deepset-ai/haystack/tree/main/rest_api
#TRANSFORMER = "intfloat/multilingual-e5-base"
#TRANSFORMER = "mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es"
TRANSFORMER = 'intfloat/multilingual-e5-large-instruct'
model = SentenceTransformer(TRANSFORMER,device="cuda")
#print(model)

description = """
Ask questions about the documents provided. 🚀
## Configs:
Model Max Sequence Length: **{}**.
Torch version: **{}**.
Is CUDA enabled? **{}**.
""".format(512, torch.__version__, torch.cuda.is_available())

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

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine", bm25_algorithm="BM25Plus")

#document_embedder = SentenceTransformersDocumentEmbedder(model=TRANSFORMER,  normalize_embeddings=True )  
#document_embedder.warm_up()
#text_embedder = SentenceTransformersTextEmbedder(model=TRANSFORMER,  normalize_embeddings=True )
#text_embedder.warm_up()
reader = ExtractiveReader(model="MMG/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es-finetuned-sqac",no_answer=False)
reader.warm_up()
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

def document_manager(files, metadata):
    docs = []
    for i,file in enumerate(files):
        nameParsed = file.filename.split(".")        
        if len(nameParsed) > 1:
            splitted_text, text_pages = ProcessText.readFile(file.file.read(),nameParsed[1])
            if len(splitted_text) == 0:
                return("error")            
            else:
                meta = None
                if metadata and len(metadata) > i:
                    meta = metadata[i]
                
                docs = docs + createDocs(splitted_text, text_pages, file.filename, meta)
                
        else:
            print("error extension1")

    if len(docs) > 0:         
        # start = time.process_time()
        # documents_with_embeddings = document_embedder.run(docs)["documents"]
        # end = time.process_time()
        # print("Processing time:",end - start)       
        document_store.write_documents(docs)
 
def createDocs(texts, pages, filename, metadata = None):
    docs = []
    if len(texts) > 0:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        
#        start = time.process_time()
        doc_emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, device="cuda", batch_size=8)#batch_size optimum for NVIDIA GeForce GTX 1650
#        end = time.process_time()
#        print("Processing time:",end - start)
        for i, text in enumerate(texts):
            currentMetadata = {'name': filename, "paragraph":i, "timestamp":dt_string}
            if metadata:
                currentMetadata.update(metadata)
            if len(pages) == len(texts):
                currentMetadata.update({"page":pages[i]})
            elif len(pages) == 1:
                currentMetadata.update({"page":pages[0]})
            
            docs.append(Document(content=text, meta=currentMetadata, embedding=doc_emb[i]))
            #docs.append(Document(content=text, meta=currentMetadata))
    return docs

def get_detailed_instruct(query: str) -> str:
    task_description = 'Given a web search query, retrieve relevant passages that answer the query'
    return f'Instruct: {task_description}\nQuery: {query}'
    
def searchInDocstore(params):
    count = params.query.strip().count(" ")+ 1
    prediction = None
    if  count < 3: 
        #print("BM25 method")
        prediction = document_store.bm25_retrieval(query=params.query.strip(), top_k=params.top_k, filters=params.filters, scale_score=True)
        for doc in prediction:
           del doc.embedding
    else:
        myquery=get_detailed_instruct(params.query.strip())
        #embedding_query = text_embedder.run(text=myquery)["embedding"]
        embedding_query = model.encode(myquery, normalize_embeddings=True, show_progress_bar=True, device="cuda", batch_size=4)
        prediction = document_store.embedding_retrieval(query_embedding=embedding_query.tolist(), top_k=params.top_k, filters=params.filters, return_embedding=False)
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
            getDocuments = document_store.filter_documents(filters={"name":doc["name"], "paragraph": context_list})
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
        
    result = reader.run(
        query=params.query.strip(),
        documents=myDocs,
        top_k=params.top_k_answers
     )["answers"]

    for answer in result:
        doc = answer.document.to_dict()
        answer.document=getContext([doc], params.context_size)
        
    return result
   
@app.post("/documents/show/", tags=["docs"])
def show_documents(filters: FilterRequest):
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
    
    prediction = document_store.filter_documents(filters=filters.filters)
    ids = [doc.id for doc in prediction]
    document_store.delete_documents(document_ids=ids)
    return True

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

        
    return {"message":"Files uploaded correctly, processing..." , "filenames": [file.filename for file in files]}