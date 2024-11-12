from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Optional, Dict

from fastapi import APIRouter, Body
import requests
from haystack.components.builders import PromptBuilder
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from openai import OpenAI

from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import RedirectResponse

from app.apis.api_documents.utils import getContext, searchInDocstore
from app.apis.api_llm.llm_utils import TEMPLATE_ASK_SINGLE_DOC, TEMPLATE_ASK_MULTIPLE_DOCS
from app.core.auth import decode_access_token
from app.core.config import EMBEDDINGS_SERVER, DB_HOST, OLLAMA_SERVER, AUDIO_TRANSCRIBE_SERVER, LLM_MODEL

router = APIRouter()

class FilterRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    token:str = Field(title="Access token", max_length=175)
    #filters: Optional[Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]] = None
    filters: Optional[Dict] = None

class SearchQueryParam(FilterRequest):
    query:str = Field(title="Query max length 1500 chars", max_length=1500)
    top_k:int = Field(default=5,gt=0, le=50, description="Number of search results")
    context_size:int = Field(default=0,ge=0, le=20, description="Number of paragraphs in context")

@dataclass
class ReturnAnswer(BaseModel):
    answer:str
    documents:list[dict]


@router.get("/", include_in_schema=False)
def main():
    return RedirectResponse(url='/docs')


@router.get("/status/")
def check_status() -> dict:
    # doc = document_store.describe_documents()
    result = {}
    try:
        res = requests.get(EMBEDDINGS_SERVER + "models/")
        res.raise_for_status()
        result.update({"embeddings_status": res.json()})
    except requests.exceptions.RequestException as e:
        result.update({"embeddings_status": "ERROR! " + str(e)})

    try:
        res = requests.get(DB_HOST + "/_cluster/health")
        res.raise_for_status()
        result.update({"db_status": res.json()})
    except requests.exceptions.RequestException as e:
        result.update({"db_status": "ERROR! " + str(e)})

    # try:
    #     res = requests.get(OLLAMA_SERVER + "api/ps")
    #     res.raise_for_status()
    #     result.update({"llm_status": res.json()})
    # except requests.exceptions.RequestException as e:
    #     result.update({"llm_status": "ERROR! " + str(e)})
    #TODO: VLLM health endpoint
    # try:
    #     res = requests.get(AUDIO_TRANSCRIBE_SERVER + "status/")
    #     res.raise_for_status()
    #     result.update({"diarization_status": res.json()})
    # except requests.exceptions.RequestException as e:
    #     result.update({"diarization_status": "ERROR! " + str(e)})

    return result


@router.post("/search/", tags=["search"])
def search_document(params: Annotated[SearchQueryParam, Body(embed=True)])->list[dict]:
    """
    Receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the system to get a set of most relevant document pieces.
    Depending on the context size, it returns previous and later paragraphs from documents

    Filter example:

    To get all documents you should provide an empty dict, like:`'{"filters": {}}`

    `{"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "name", "operator": "==", "value": "2019"},
        			{"field": "companies", "operator": "in", "value": ["BMW", "Mercedes"]}
      					]
    			   }
  		      }`
    """
    project_index = decode_access_token(params.token)
    print(project_index)
    document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search")
    docs = searchInDocstore(params, document_store)
    print(docs)
    result = []
    if docs is None:
        return result

    for doc in docs:
        result.append(doc.to_dict())
    docs_context = getContext(result, params.context_size, document_store)
    document_store.client.close()
    return docs_context


@router.post("/ask/", tags=["search"])
def ask_document(params: Annotated[SearchQueryParam, Body(embed=True)]) -> ReturnAnswer:
    """
    Receive the question as a string and allows the requester to set
    additional parameters that will be passed on to the system to get a set of most relevant answers.

    Filter example:

    To get all documents you should provide an empty dict, like:`'{"filters": {}}`

    `{"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "name", "operator": "==", "value": "2019"},
        			{"field": "companies", "operator": "in", "value": ["BMW", "Mercedes"]}
      					]
    			   }
  		      }`
    """
    project_index = decode_access_token(params.token)
    print(project_index)
    document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search")
    myDocs = searchInDocstore(params, document_store)
    if myDocs is None:
        #return {"answer": "None", "documents": []}
        return ReturnAnswer(answer="None",documents=[])

    result = []
    for doc in myDocs:
        result.append(doc.to_dict())
    docs_context = getContext(result, params.context_size, document_store,include_paragraphs=True)

    #prepara el texto para enviar al llm
    grouped_docs = {}
    #Agrupa documentos por nombre incluyendo el contexto
    for doc in docs_context:
        name = doc["name"]
        # Si el nombre no está en el diccionario, lo agrega con un contenido vacío
        if name not in grouped_docs:
            grouped_docs[name] = []
        # Agrega el contenido al arreglo correspondiente al nombre
        grouped_docs[name].append((doc["content"], doc["paragraph"]))
        if "before_context" in doc and "after_context" in doc:
            grouped_docs[name].extend(doc["before_context"])
            grouped_docs[name].extend(doc["after_context"])

    for name, content in grouped_docs.items():
        tuplas_sin_duplicados = list(set(content))  #elimina parrafos duplicados
        ordenadas = sorted(tuplas_sin_duplicados, key=lambda x: x[1])   #ordena de menor a mayor los parrafos
        list_temp= []
        #agrupa por parrafos contiguos
        for i, value in enumerate(ordenadas):
            if i == 0 or (value[1] - 1 != ordenadas[i - 1][1]):
                list_temp.append(value[0])
            else:
                list_temp[-1] += value[0]

        grouped_docs[name] =  list_temp


    builder = PromptBuilder(template=TEMPLATE_ASK_MULTIPLE_DOCS)
    my_prompt = builder.run(myDocs=grouped_docs, query=params.query.strip())["prompt"].strip()

    client = OpenAI(base_url=OLLAMA_SERVER + 'v1/', api_key='ollama', )
    response = client.chat.completions.create(
        messages=[dict(content=my_prompt, role="user")],
        model=LLM_MODEL,
        temperature=0.1,
        top_p=0.5,
        stream=False
    )

    #remove t-uplas from contexts arrays
    for doc in docs_context:
        if "before_context" in doc and "after_context" in doc:
            doc["before_context"] = [item[0] for item in doc["before_context"]]
            doc["after_context"] = [item[0] for item in doc["after_context"]]

    #final_result = {"answer": response.choices[0].message.content, "documents": docs_context}
    final_result= ReturnAnswer(answer=response.choices[0].message.content, documents=docs_context)
    document_store.client.close()
    return final_result


@router.post("/show/", tags=["documents"])
def show_documents(filters: FilterRequest)->list[dict]:
    """
    Retrieve documents contained in your document store.
    You can filter the documents to retrieve by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Filter example:

    To get all documents you should provide an empty dict, like:`'{"filters": {}}`

    `{"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "name", "operator": "==", "value": "2019"},
        			{"field": "companies", "operator": "in", "value": ["BMW", "Mercedes"]}
      					]
    			   }
  		      }`
    """
    project_index = decode_access_token(filters.token)
    print(project_index)
    document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search")
    prediction = document_store.filter_documents(filters=filters.filters)
    result = []
    for doc in prediction:
        doc_dict = doc.to_dict()
        del doc_dict["embedding"]
        result.append(doc_dict)
    document_store.client.close()
    # print_documents(prediction, max_text_len=100, print_name=True, print_meta=True)
    return result


@router.post("/name_list/", tags=["documents"])
def list_documents_name(filters: FilterRequest)-> list[str]:
    """
       Retrieve the filename of all documents loaded.
       You can filter the documents to retrieve by metadata (like the document's name),
       or provide an empty JSON object to clear the document store.

       Filter example:

       To get all documents you should provide an empty dict, like:`'{"filters": {}}`

       `{"filters": { "operator": "AND",
                   "conditions": [
                       {"field": "name", "operator": "==", "value": "2019"},
                       {"field": "companies", "operator": "in", "value": ["BMW", "Mercedes"]}
                           ]
                      }
                 }`
   """
    # print(filters.filters)
    project_index = decode_access_token(filters.token)
    print(project_index)
    document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search")
    prediction = document_store.filter_documents(filters=filters.filters)

    res = []
    for doc in prediction:
        res.append(doc.meta["name"])
    res = list(set(res))
    document_store.client.close()
    return res


@router.post("/delete/", tags=["documents"])
def delete_documents(filters: FilterRequest) -> bool:
    """
    Delete documents contained in your document store.
    You can filter the documents to delete by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Filter example:

    To get all documents you should provide an empty dict, like:`'{"filters": {}}`

    `{"filters": { "operator": "AND",
      			"conditions": [
        			{"field": "name", "operator": "==", "value": "2019"},
        			{"field": "companies", "operator": "in", "value": ["BMW", "Mercedes"]}
      					]
    			   }
  		      }`
    """
    project_index = decode_access_token(filters.token)
    print(project_index)
    document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search")
    prediction = document_store.filter_documents(filters=filters.filters)
    ids = [doc.id for doc in prediction]
    document_store.delete_documents(document_ids=ids)
    document_store.client.close()
    return True