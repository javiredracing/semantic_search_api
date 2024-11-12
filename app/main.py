from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.core import auth
from app.routes import views, views_upload, views_agents

description = """Ask questions about the documents provided. 🚀
Model Max Sequence Length: **512**.
Services required: 
* infinity service (https://github.com/michaelfeil/infinity)
* Elastic search service.
* vLLM service.
"""

tags_metadata = [
    {
       "name": "projects",
       "description": "Manage token's projects",
    },
    {
        "name": "search",
        "description": "Get answers into documents",
        #"externalDocs": {
        #    "description": "Made with FastApi",
        #    "url": "https://fastapi.tiangolo.com/",
        #},
    },
    {
        "name": "documents",
        "description": "Manage documents",
    },
    {
        "name": "agents",
        "description": "Helpful agents",
    },
]

app = FastAPI(
    title="Semantic search service",
    description=description,
    summary="Semantic search in documents",
    version="3.0.0",
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

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(views.router)
app.include_router(views_upload.router)
app.include_router(views_agents.router)