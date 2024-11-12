import requests
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever, \
    ElasticsearchEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

from app.core.config import EMBEDDINGS_SERVER, LLM_MODEL


def get_detailed_instruct(query: str) -> str:
    task_description = 'Given a web search query, retrieve relevant passages that answer the query'
    return f'Instruct: {task_description}\nQuery: {query}'

def searchInDocstore(params, document_store:ElasticsearchDocumentStore):
    count = params.query.strip().count(" ")+ 1
    prediction = None

    if  count < 3:
        #print("BM25 method")
        bm25retriever = ElasticsearchBM25Retriever(document_store=document_store, scale_score=True)
        prediction = bm25retriever.run(query=params.query.strip(),top_k=params.top_k, filters=params.filters)["documents"]
        #prediction = document_store.bm25_retrieval(query=params.query.strip(), top_k=params.top_k, filters=params.filters, scale_score=True)
        for doc in prediction:
           del doc.embedding
    else:
        myquery=get_detailed_instruct(params.query.strip())
        #embedding_query = text_embedder.run(text=myquery)["embedding"]
        #embedding_query = model.encode(myquery, normalize_embeddings=True, show_progress_bar=True, device="cuda", batch_size=4)
        try:
            req = requests.post(EMBEDDINGS_SERVER+"embeddings", json={"input": myquery,"model": LLM_MODEL})    #encode query
            if req.status_code == 200:
                query_embedding = req.json()["data"][0]["embedding"]
                #prediction = document_store.embedding_retrieval(query_embedding=query_embedding, top_k=params.top_k, filters=params.filters, return_embedding=False)
                retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)
                prediction = retriever.run(query_embedding=query_embedding, top_k=params.top_k, filters=params.filters)["documents"]
                for doc in prediction:
                   del doc.embedding
        except requests.exceptions.RequestException as e:
            return None
    #print_documents(prediction, max_text_len=query.max_text_len, print_name=True, print_meta=True)
    return prediction

def getContext(docs, context_size:int, document_store:ElasticsearchDocumentStore, include_paragraphs:bool = False) -> []:
    if context_size > 0:
        for doc in docs:
            currentParagraph = doc["paragraph"]
            min_paragrah = currentParagraph - context_size
            if min_paragrah < 0:
                min_paragrah = 0
            min_paragrah = list(range(min_paragrah, currentParagraph))
            max_paragraph = list(range(currentParagraph + 1, currentParagraph + context_size + 1))
            context_list = min_paragrah + max_paragraph
            current_filters = {"operator": "AND",
                               "conditions": [{"field": "name", "operator": "==", "value": doc["name"]},
                                              {"field": "paragraph", "operator": "in", "value": context_list}, ]}
            getDocuments = document_store.filter_documents(filters=current_filters)
            doc["before_context"] = []
            doc["after_context"] = []
            for context in getDocuments:
                if context.meta["paragraph"] < currentParagraph:
                    if include_paragraphs:
                        doc["before_context"].append((context.content, context.meta["paragraph"]))
                    else:
                        doc["before_context"].append(context.content)
                else:
                    if include_paragraphs:
                        doc["after_context"].append((context.content, context.meta["paragraph"]))
                    else:
                        doc["after_context"].append(context.content)
    return docs