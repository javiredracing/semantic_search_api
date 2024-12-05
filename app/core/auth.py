from __future__ import annotations

from datetime import datetime, timedelta
from http import HTTPStatus
import logging
from fastapi import APIRouter, HTTPException
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from jose import jwt, JWTError
from ldap3 import Server, Connection
from pydantic import BaseModel, SecretStr, Field

from app.core import config
from app.core.config import LDAP_dc, LDAP_ou, API_USERNAME, API_PASSWORD, LDAP_server, DB_HOST, PROJECTS_INDEX, \
    INDEX_SETTINGS, ADMIN_USERS


class Project(BaseModel):
    project: str
    token: str

class ProjectInfo(Project):
    user:str
    description:str

class Login(BaseModel):
    username:str= Field(title="Max length 50 chars", max_length=50)
    password:SecretStr = Field(title="Max length 50 chars", max_length=50)

class ProjectParams(Login):
    project:str= Field(title="Project name", max_length=20)
    description:str= Field(title="Project's description", max_length=1000)

router = APIRouter()

def authenticate_superuser(username:str,password:str) -> bool:
    return username == API_USERNAME and password == API_PASSWORD


def authenticate_user(username: str, password: str) -> bool:
    username_mod = f'uid={username},ou={LDAP_ou},dc=iter,dc=es'
    server = Server(LDAP_server, use_ssl=False, get_info='ALL')
    conn = Connection(server, username_mod, password, auto_bind=False, auto_referrals=False, raise_exceptions=False)
    result = False
    try:
        conn.bind()
        conn.password = None
        if conn.result['result'] != 0:
            result = False
        else:
            result = True

    except Exception as e:
        logging.error(str(e).replace(LDAP_dc, 'server'))
    finally:
        if conn.bound: conn.unbind()

    return result


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # access_token_expires = timedelta(
        #     minutes=config.API_ACCESS_TOKEN_EXPIRE_MINUTES,
        # )
        #expire = datetime.utcnow() + timedelta(minutes=15)
        #expire = datetime.utcnow() + access_token_expires
        expire = datetime(2035, 1, 1)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        config.API_SECRET_KEY,
        algorithm=config.API_ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> str:
    credentials_exception = HTTPException(
        status_code=HTTPStatus.UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    try:
        payload = jwt.decode(
            token,
            config.API_SECRET_KEY,
            algorithms=[config.API_ALGORITHM],
        )
        project_index = payload.get("sub")

        if project_index is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    return project_index


@router.post("/list/",  tags=["projects"])
async def list_projects(login:Login) -> list[ProjectInfo]:
    """
    List all user's projects. Parameters required:
    - username: Valid company username
    - password: username password
    """
    username = login.username.lower().strip()
    isSuperuser = authenticate_superuser(username, login.password.get_secret_value())
    if not isSuperuser:
        if not authenticate_user(username, login.password.get_secret_value()):
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail="Incorrect username or password",
            )

    try:
        document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index=PROJECTS_INDEX, custom_mapping=INDEX_SETTINGS)
        query = {"query": {"match": {"username": username}}}
        if isSuperuser or username in ADMIN_USERS:
            query = {"query": {"match_all": {}}}

        response = document_store.client.search(index=PROJECTS_INDEX, body=query)
        result = []
        document_store.client.close()
        for hit in response['hits']['hits']:
            project = hit['_source']["project"]
            user_project = hit['_source']["user"]
            my_token= create_access_token(data={"sub": project,"user":user_project})
            info=ProjectInfo(project=project,user=user_project,token=my_token,description= hit['_source']["description"])
            result.append(info)
        return result

    except Exception as e:
        logging.error(e)
        raise HTTPException(
            status_code=HTTPStatus.FAILED_DEPENDENCY,
            detail="Error in database",
        )

@router.post("/create/", response_model=Project, tags=["projects"])
async def login_for_access_token(params:ProjectParams) -> Project:
    """
    Create a new project. Valid parameters:
    - username: Valid company username
    - password: username password
    - project: Project's name
    - Description
    """
    username = params.username.lower().strip()
    isSuperuser = authenticate_superuser(username, params.password.get_secret_value())
    if not isSuperuser:
        if not authenticate_user(username, params.password.get_secret_value()):
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail="Incorrect username or password",
            )

    project = params.project.lower().strip().replace(" ", "_")
    if project == PROJECTS_INDEX:
        project = "_" + project
    try:
        document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index=PROJECTS_INDEX, custom_mapping=INDEX_SETTINGS)
        response = document_store.client.search(index=PROJECTS_INDEX, body={"query": {"match": {"project": project}}})
        if response['hits']['total']['value'] > 0:
            document_store.client.close()
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail="Project already exists!",
            )
        else:
            document_store.client.index(index=PROJECTS_INDEX,
                                        body={"project": project, "description": params.description, "user": username})
            document_store.client.close()
            document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index=project)
            document_store.client.close()
    except Exception as e:
        #print(e)
        logging.error(e)
        raise HTTPException(
            status_code=HTTPStatus.FAILED_DEPENDENCY,
            detail="Error in database",
        )

    access_token = create_access_token(
        data={"sub": project, "user":username}
    )

    return Project(project=project, token=access_token)