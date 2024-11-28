import logging

import requests
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever, \
    ElasticsearchEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack import Document
from app.core.config import EMBEDDINGS_SERVER, EMBEDDINGS_MODEL, RERANKER_MODEL


def get_detailed_instruct(query: str) -> str:
    if EMBEDDINGS_MODEL == "intfloat/multilingual-e5-large-instruct":
        task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        return f'Instruct: {task_description}\nQuery: {query}'
    else:
        return query

def searchInDocstore(params, document_store:ElasticsearchDocumentStore) -> list[dict] | None:
    count = params.query.strip().count(" ") + 1
    prediction = None

    #print("BM25 method")
    bm25retriever = ElasticsearchBM25Retriever(document_store=document_store, scale_score=True)
    prediction = bm25retriever.run(query=params.query.strip(),top_k=params.top_k, filters=params.filters)["documents"]
    #prediction = document_store.bm25_retrieval(query=params.query.strip(), top_k=params.top_k, filters=params.filters, scale_score=True)

    if count < 3:
        myquery=get_detailed_instruct(params.query.strip())
        #embedding_query = text_embedder.run(text=myquery)["embedding"]
        #embedding_query = model.encode(myquery, normalize_embeddings=True, show_progress_bar=True, device="cuda", batch_size=4)
        try:
            req = requests.post(EMBEDDINGS_SERVER+"embeddings", json={"model": EMBEDDINGS_MODEL, "input": myquery})    #encode query
            req.raise_for_status()
            query_embedding = req.json()["data"][0]["embedding"]
            #prediction = document_store.embedding_retrieval(query_embedding=query_embedding, top_k=params.top_k, filters=params.filters, return_embedding=False)
            retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)
            prediction_retriever = retriever.run(query_embedding=query_embedding, top_k=params.top_k, filters=params.filters)["documents"]
            #join sparse and embeddings retrieved result with reranker
            joiner = DocumentJoiner(join_mode="concatenate")
            prediction = joiner.run(documents=[prediction, prediction_retriever])["documents"]
        except requests.exceptions.RequestException as e:
            logging.error(e.response.text)
            return None

    for doc in prediction:
       del doc.embedding
    prediction = rerank(query=params.query.strip(), docs=prediction, top_k=params.top_k)
    #print_documents(prediction, max_text_len=query.max_text_len, print_name=True, print_meta=True)

    list_json = []
    for json_doc in prediction:
        json_item = json_doc.to_dict()
        json_item["before_context"] = []
        json_item["after_context"] = []
        list_json.append(json_item)

    return list_json

def set_context(docs:list[dict], context_size:int, document_store:ElasticsearchDocumentStore, include_paragraphs:bool = False) -> list[dict]:
    if context_size > 0:
        for doc in docs:
            currentParagraph = doc["paragraph"]
            min_paragraph = currentParagraph - context_size
            if min_paragraph < 0:
                min_paragraph = 0
            min_paragraph = list(range(min_paragraph, currentParagraph))
            max_paragraph = list(range(currentParagraph + 1, currentParagraph + context_size + 1))
            context_list = min_paragraph + max_paragraph
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

def rerank(query:str, docs:list[Document], top_k:int) -> list[Document]:
    if len(docs) > 0:
        corpus:list[str] = [doc.content for doc in docs]
        #TODO include context in corpus
        try:
            req = requests.post(EMBEDDINGS_SERVER + "rerank", json={"model": RERANKER_MODEL, "query": query.strip(), "documents": corpus})
            req.raise_for_status()
            for result in req.json()["results"]:
                docs[result["index"]].score = result['relevance_score']
            reordered_list = sorted(docs, key=lambda x: x.score, reverse=True)
            return reordered_list[:top_k] if len(docs) >= top_k else reordered_list
        except requests.exceptions.RequestException as e:
            logging.error(e.response.text)
    return docs