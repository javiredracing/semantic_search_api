# semantic_search_api
- upload documents/audio
- show documents
- ask_documents
- delete documents
- summarize documents


# launch:
- fastapi run main.py

# docker
- build: sudo docker build -t  hub.iter.es/semantic_search_api:V0.2 .
- dev: sudo docker run  -it --network host -v /home/administrador/audio2:/home/administrador/audio2 --name semantic_search_api hub.iter.es/semantic_search_api:V0.2
- pro: sudo docker run  -d --network host -v /home/administrador/audio2:/home/administrador/audio2 --name semantic_search_api hub.iter.es/semantic_search_api:V0.2