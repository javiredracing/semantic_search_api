import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.api_a.mainmod import main_func as main_func_a
from app.api_b.mainmod import main_func as main_func_b
from app.core import config

router = APIRouter()
security = HTTPBasic()


def authorize(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, config.api_username)
    correct_password = secrets.compare_digest(credentials.password, config.api_password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


@router.get("/api-a/{num}", tags=["api_a"])
async def views_a(num: int, auth=Depends(authorize)):
    if auth is True:
        return main_func_a(num)


@router.get("/api-b/{num}", tags=["api_b"])
async def views_b(num: int, auth=Depends(authorize)):
    if auth is True:
        return main_func_b(num)
