from __future__ import annotations

from fastapi import APIRouter, HTTPException
from haystack.components.builders import PromptBuilder
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from openai import OpenAI
from starlette.responses import PlainTextResponse

from app.apis.api_llm.llm_utils import TEMPLATE_SUMMARY, TEMPLATE_TRANSLATE
from app.core.auth import decode_access_token
from app.core.config import DB_HOST, OLLAMA_SERVER, LLM_MODEL

router = APIRouter()

@router.get("/agents/summarize/{token}/{file_name}/", tags=["agents"], response_class=PlainTextResponse)
def get_summary(file_name: str, token:str):
    """
    Devuelve un resumen que captura los puntos más importantes.
    """
    index = decode_access_token(token)
    print(index)
    filters = {"field": "name", "operator": "==", "value": file_name.strip()}
    document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search") #TODO: Replace index
    prediction = document_store.filter_documents(filters=filters)
    document_store.client.close()

    builder = PromptBuilder(template=TEMPLATE_SUMMARY)

    #options = {"num_predict": -1, "temperature": 0.1}

    if len(prediction) > 0:
        orderedlist = sorted(prediction, key=lambda doc: doc.meta['paragraph'])  # order by paragraph
        custom_docs = ""
        current_speaker = ""
        response = []
        client = OpenAI(base_url=OLLAMA_SERVER + 'v1/', api_key='ollama', )
        for doc in orderedlist:
            if doc.meta["file_type"].startswith("audio/"):
                if doc.meta['speaker'] != current_speaker:
                    if current_speaker != "" and not custom_docs.strip().endswith((".", ",", "!", "?", ";")):
                        custom_docs = custom_docs.strip() + "..."
                    custom_docs += f"\n\n- {doc.meta['speaker']}: "
                    current_speaker = doc.meta['speaker']
            else:
                if not custom_docs.strip().endswith((".", ",", "!", "?", ";")) and doc.meta['paragraph'] != 0:
                    custom_docs = custom_docs.strip() + "..."
                custom_docs += "\n\n"

            custom_docs += f"{doc.content} "

            if len(custom_docs) > 3500:
                if doc.meta["file_type"].startswith("audio/"):
                    current_speaker = ""
                my_prompt = builder.run(myDocs=custom_docs)["prompt"].strip()
                # print(my_prompt)
                custom_docs = ""
                #LLM CALL
                completion = client.chat.completions.create(
                    messages=[dict(content=my_prompt, role="user")],
                    model=LLM_MODEL,
                    temperature=0.1,
                    top_p=0.5,
                    stream=False
                )

                response.append("\n\n" + completion.choices[0].message.content)

        if not custom_docs.strip().endswith((".", ",", "!", "?", ";")):
            custom_docs = custom_docs.strip() + "."
        if len(custom_docs) > 0:
            my_prompt = builder.run(myDocs=custom_docs)["prompt"].strip()
            custom_docs = ""
            completion = client.chat.completions.create(
                messages=[dict(content=my_prompt, role="user")],
                model=LLM_MODEL,
                temperature=0.1,
                top_p=0.5,
                stream=False
            )
            #if agent_response is None:
            #    return ""
            response.append("\n\n" + completion.choices[0].message.content)

        return ''.join(response)
    else:
        raise HTTPException(status_code=404, detail="File not found!")


@router.get("/audio/get_SRT_translated/{token}/{file_name}/{lang}/", tags=["audio"], response_class=PlainTextResponse)
def get_SRT_traslated(file_name: str, lang: str, token:str):
    """
    Get the srt file referred to a specific audio file that have been processed previously, translated into a defined language. The language param admit most commons languages.
    Ej: *Español, inglés, alemán, italiano, francés,...*
    """
    index = decode_access_token(token)
    print(index)
    document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index="semantic_search")  # TODO: Replace index
    filters = {"field": "name", "operator": "==", "value": file_name.strip()}
    prediction = document_store.filter_documents(filters=filters)
    document_store.client.close()
    if len(prediction) > 0:
        orderedlist = sorted(prediction, key=lambda d: d.meta['paragraph'])  # order by paragraph
        # print("total docs",len(orderedlist))

        builder = PromptBuilder(template=TEMPLATE_TRANSLATE)
        #options = {"num_predict": -1, "temperature": 0.1, "top_p": 0.5}

        prompt_list = []
        srt_file = ""
        count = 0
        for i, doc in enumerate(orderedlist):
            if doc.meta["file_type"].startswith("audio/"):
                my_doc_mod = doc.content
                srt_file += f"{str(i)}- {my_doc_mod.capitalize()}\n\n"
                count += len(doc.content)
                if count >= 1500:  # 6000 chars is aprox 1500 tokens
                    my_prompt = builder.run(lang=lang, srt_file=srt_file)["prompt"].strip()
                    prompt_list.append(my_prompt)
                    srt_file = ""
                    count = 0

        if len(srt_file) > 0:
            my_prompt = builder.run(lang=lang, srt_file=srt_file)["prompt"].strip()
            prompt_list.append(my_prompt)

        srt_file = []
        client = OpenAI(base_url=OLLAMA_SERVER + 'v1/', api_key='ollama', )
        for prompt in prompt_list:
            completion = client.chat.completions.create(
                messages=[dict(content=prompt, role="user")],
                model=LLM_MODEL,
                temperature=0.1,
                top_p=0.5,
                stream=False
            )
            data = completion.choices[0].message.content.split("\n")
            # print(data)
            final_list = [i for i in data if i]
            for item in final_list[1:]:
                x = item.find("-")
                if x >= 1 and (x + 1) < len(item):
                    srt_file.append(item[(x + 1):].strip())

        #        print("total traslated paragraphs",len(srt_file))
        plain_res = ""
        for i, doc in enumerate(orderedlist):
            if doc.meta["file_type"].startswith("audio/"):
                plain_res += f"{str(i + 1)}\n"
                text1 = f"{doc.meta['start_time']} --> {doc.meta['end_time']}\n"
                plain_res += text1
                #text1 = f"{doc.meta['speaker']}: {doc.content}\n"
                #plain_res += text1
                if i < len(srt_file):
                    text1 = f"{doc.meta['speaker']}: {srt_file[i]}\n\n"
                else:
                    text1 = f"{doc.meta['speaker']}: --\n\n"
                plain_res += text1

        return plain_res
    else:
        raise HTTPException(status_code=404, detail="File not found!")