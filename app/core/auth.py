from __future__ import annotations

from datetime import datetime, timedelta
from http import HTTPStatus

from fastapi import APIRouter, Depends, HTTPException
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from jose import jwt, JWTError
from ldap3 import Server, Connection
from pydantic import BaseModel, SecretStr
from starlette.datastructures import Secret

from app.core import config
from app.core.config import LDAP_dc, LDAP_ou, API_USERNAME, API_PASSWORD, LDAP_server, DB_HOST, PROJECTS_INDEX, \
    INDEX_SETTINGS


class Token(BaseModel):
    access_token: str
    project: str


router = APIRouter()

def authenticate_superuser(username:str,password:str):
    return username.lower().strip() == API_USERNAME and password == API_PASSWORD


def authenticate_user(
    username: str,
    password: str,
) -> bool:
    username_mod = f'uid={username},ou={LDAP_ou},dc=iter,dc=es'
    # conn = Connection(server, 'uid=jfernandez,ou=users,dc=iter,dc=es', password, auto_bind=False, auto_referrals=False, raise_exceptions=False)
    server = Server(LDAP_server, use_ssl=False, get_info='ALL')
    # print("server",username_mod)
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
        print(str(e).replace(LDAP_dc, 'server'))
    finally:
        if conn.bound: conn.unbind()

    return result


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

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


@router.post("/create", response_model=Token)
async def login_for_access_token(
    username:str, password:SecretStr, project:str, description:str
) -> dict[str, str]:
    username = username.lower().strip()
    user = authenticate_user(
        username,
        password.get_secret_value(),
    )

    if not user:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    project = project.lower().strip()
    document_store = ElasticsearchDocumentStore(hosts=DB_HOST, index=PROJECTS_INDEX, custom_mapping=INDEX_SETTINGS)
    try:
        #response = document_store.client.search(index=PROJECTS_INDEX, body={"query": {"match": {"project": project}}})
        if document_store.client.exists(index=project):
            document_store.client.close()
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail="Project already exists!",
            )
        else:
            document_store.client.index(index=PROJECTS_INDEX,
                                        body={"project": project, "description": description, "user": username})
    except Exception as e:
        document_store.client.close()
        raise HTTPException(
            status_code=HTTPStatus.FAILED_DEPENDENCY,
            detail="Error in database",
        )

    document_store.client.close()
    access_token_expires = timedelta(
        seconds=config.API_ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    access_token = create_access_token(
        data={"sub": project},  # type: ignore
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "project": project}
