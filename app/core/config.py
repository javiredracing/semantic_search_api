import pathlib

from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings

ROOT = pathlib.Path(__file__).resolve().parent.parent  # app/
BASE_DIR = ROOT.parent  # ./

config = Config(BASE_DIR / ".env")


API_USERNAME = config("API_USERNAME", str)
API_PASSWORD = config("API_PASSWORD", str)

# Auth configs.
API_SECRET_KEY = config("API_SECRET_KEY", str)
API_ALGORITHM = config("API_ALGORITHM", str)
API_ACCESS_TOKEN_EXPIRE_MINUTES = config(
    "API_ACCESS_TOKEN_EXPIRE_MINUTES", int
)  # infinity

DB_HOST = config("DATABASE_SERVER", str)
EMBEDDINGS_SERVER = config("EMBEDDINGS_SERVER",str)
OLLAMA_SERVER = config("OLLAMA_SERVER", str)
AUDIO_TRANSCRIBE_SERVER = config("AUDIO_TRANSCRIBE_SERVER",str)
AUDIO_PATH = config("AUDIO_PATH",str)

INDEX_SETTINGS = {'mappings':
                      {'_source': {'enabled': True},
                       'properties':
                           {'project':{'type': 'text'},
                            'description': {'type': 'text'},
                            'user': {'type': 'text'}
                            }
                       }
                  }

PROJECTS_INDEX = "projects_index"

LDAP_ou = config("LDAP_ou",str)
LDAP_dc = config("LDAP_dc",str)
LDAP_server = config("LDAP_server",str)
ADMIN_USERS = config("ADMIN_USERS", cast=CommaSeparatedStrings)

LLM_MODEL = config("LLM_MODEL",str)
EMBEDDINGS_MODEL = config("TRANSFORMER",str)