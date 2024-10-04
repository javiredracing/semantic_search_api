from __future__ import annotations

from typing import Annotated, Optional, Dict

from fastapi import APIRouter, Body
import requests
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import RedirectResponse

from app.apis.api_documents.utils import getContext, searchInDocstore
from app.core.auth import decode_access_token
from app.core.config import EMBEDDINGS_SERVER, DB_HOST, OLLAMA_SERVER, AUDIO_TRANSCRIBE_SERVER

router = APIRouter()

class FilterRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    token:str = Field(title="Access token", max_length=175)
    #filters: Optional[Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]] = None
    filters: Optional[Dict] = None

class SearchQueryParam(FilterRequest):
    query:str = Field(title="Query max length 1500 chars", max_length=1500)
    top_k:int = Field(default=5,gt=0, le=50, description="Number of search results")
    context_size:int = Field(default=0,ge=0, le=20, description="Number of paragrhaps in context")

@router.get("/", include_in_schema=False)
def main():
    return RedirectResponse(url='/docs')

@router.get("/status/{audio_file}", tags=["audio"])
async def get_audio_status(audio_file: str) -> dict:
    '''Show status of a file audio in the transcription batch processing.
    * audio_file is the current audio that may be in process
    '''
    result = {
        "audio_file": audio_file,
        "status": "ERROR!"
    }
    try:
        res = requests.get(AUDIO_TRANSCRIBE_SERVER + "status/" + audio_file)
        res.raise_for_status()
        result = res.json()
    except requests.exceptions.RequestException as e:
        result.update({"status": "ERROR! " + str(e)})

    return result

@router.get("/status/")
def check_status():
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

    try:
        res = requests.get(OLLAMA_SERVER + "api/ps")
        res.raise_for_status()
        result.update({"llm_status": res.json()})
    except requests.exceptions.RequestException as e:
        result.update({"llm_status": "ERROR! " + str(e)})

    try:
        res = requests.get(AUDIO_TRANSCRIBE_SERVER + "status/")
        res.raise_for_status()
        result.update({"diarization_status": res.json()})
    except requests.exceptions.RequestException as e:
        result.update({"diarization_status": "ERROR! " + str(e)})

    return result
#
@router.post("/search/", tags=["search"])
def search_document(params: Annotated[SearchQueryParam, Body(embed=True)]):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the system to get a set of most relevant document pieces.
    Depending on the context size, it return previous and later paragraphs from documents

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

    result = []
    if docs is None:
        return result

    for doc in docs:
        result.append(doc.to_dict())
    docs_context = getContext(result, params.context_size, document_store)
    document_store.client.close()
    return docs_context