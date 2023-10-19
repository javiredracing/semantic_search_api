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
import fitz
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
INPUT_FORMATS_FITZ = ["pdf", "xps", "epub", "mobi", "fb2", "cbz", "svg"]
VALID_FORMATS = ["txt"]
#https://github.com/deepset-ai/haystack/tree/main/rest_api
TRANSFORMER = "intfloat/multilingual-e5-small"
#TRANSFORMER = "mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es"
model = SentenceTransformer(TRANSFORMER)
#print(model)

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
reader = FARMReader("MMG/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es-finetuned-sqac", use_gpu=True) 
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
    query:str = Field(title="Query max length 512 chars", max_length=512)
    #max_text_len:int = Field(default=1000 ,gt=0, le=1500, description="Max length of results text")
    top_k:int = Field(default=5,gt=0, le=50, description="Number of search results")
    
class AskQueryParams(SearchQueryParam):
    top_k_answers:int = Field(default=2,gt=0, le=50, description="Number of ask results")

def remove_header_footer(text, header=0, footer=0):
    text_splitted = text.splitlines()
    result = ""
    value = len(text_splitted)
    if footer > 0:
        value = -1 * footer
    for line in text_splitted[header:value]:
        result = result + line + "\r\n"
    result = result + "\f\r\n"
    return result
    
def readFile(file, header =0, footer=0):
    text = file.file.read().decode("utf-8")
    if header > 0 or footer > 0:
        pages = text.split("\f")
        text_formatted = ""
        for page in pages:
            text_formatted = text_formatted + remove_header_footer(page, header, footer)
        return text_formatted
    else:
        return text
    
def readFileFitz(file, format, header, footer):
    #TODO: https://towardsdatascience.com/extracting-text-from-pdf-files-with-python-a-comprehensive-guide-9fc4003d517
    bytesFile = file.file.read()
    doc = fitz.open(stream=bytesFile, filetype=format)
    pdf_formatted = ""
    for page in doc: # iterate the document pages
        text = page.get_text()#.encode("utf8") # get plain text (is in UTF-8)
        text = remove_header_footer(text, header, footer)
        pdf_formatted = pdf_formatted + text
    return pdf_formatted

def document_manager(files, metadata, force_update, parse_doc, header, footer):
    docs = []
    for i,file in enumerate(files):
        nameParsed = file.filename.split(".")
        if len(nameParsed) > 1:            
            text = ""
            if nameParsed[1].lower() in INPUT_FORMATS_FITZ:
                text = readFileFitz(file, nameParsed[1].lower(), header, footer)
            elif nameParsed[1].lower() in VALID_FORMATS:
                text = readFile(file, header, footer)
            else:
                print("Error extension")
                break
                
            splitted_text = []
            text_pages = []
            if parse_doc:
                splitted_text, text_pages = processText(text)
            else:
                splitted_text.append(text.strip())
                text_pages.append(1)
            text = ""    

            if len(splitted_text) > 0:
                if force_update:
                    document_store.delete_documents(filters={'name': file.filename})
                meta = None
                if metadata and len(metadata) > i:
                    meta = metadata[i]
                docs = docs + generateEmbeddings(splitted_text, text_pages, file.filename, meta)
                
        else:
            print("error extension")
            #break

    if len(docs) > 0:    
        document_store.write_documents(docs)

def processText(input):
    '''
    parse text by empty line. clean emtpy lines and whites in the end.
    '''
    text = ""
    splitted_text = []
    text_page = []
    buf = io.StringIO(input)
    file_content = buf.readlines()
    page_counter = 1
    for line in file_content:
        #line = line.decode("utf-8")       
        aux = line.strip()
        if len(aux) == 0 :  #breakline
            if len(text) > 0 and text.strip().endswith("."):
                splitted_text.append(text.strip())                
                text_page.append(page_counter)
                text = ""
            if "\f" in line:    #end of page
                page_counter+=1
            #continue
        else:
            if not aux.endswith(".") and not aux.endswith(":"):
                line = aux + " "
            text = text + line

    return splitted_text, text_page
    
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
    additional parameters that will be passed on to the system to get a set of most relevant document pieces
    
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
    return searchInDocstore(params)

    
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
    #print_answers(result, details="all") 
    return result
   
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
def upload_documents(files: Annotated[List[UploadFile], File(description="TXT files")], background_tasks: BackgroundTasks, metadata:InputParams = Body(None) , force_update:Annotated[bool, Form(description="Remove older files")] = True, parse_doc:Annotated[bool, Form(description="Parse doc in paragrahps")] = True, header: Annotated[int,Form(description="Header lines to remove")] = 0, footer: Annotated[int, Form(description="Footer lines to remove")] = 0):
#def upload_documents(files: Annotated[List[UploadFile], File()], metadata: InputParams = Body(...)):
    """
    This endpoint accepts documents in .pdf .xps, .epub, .mobi, .fb2, .cbz, .svg and .txt format at this time. It only can be parsed by "\\n" character.
    
    `TODO: Add support for other document formats (DOC,...) and more parsing options.`
    
    Metadata example for each file added: `{"name": ["some", "more"], "category": ["only_one"]}`    OR    empty
    """
    #
    #https://stackoverflow.com/questions/63110848/how-do-i-send-list-of-dictionary-as-body-parameter-together-with-files-in-fastap
    

    background_tasks.add_task(document_manager, files, metadata.metadata, force_update, parse_doc, header, footer)

    return {"message":"Files uploaded correctly, processing..." , "filenames": [file.filename for file in files]}