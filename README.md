<div align="center">

## Features

-   It uses [FastAPI][fastapi] framework for API development. FastAPI is a modern, highly
    performant, web framework for building APIs with Python 3.6+.

-   The APIs are served with [Gunicorn](gunicorn) server with multiple [Uvicorn][uvicorn]
    workers. Uvicorn is a lightning-fast "ASGI" server. It runs asynchronous Python web code
    in a single process.

-   Simple reverse-proxying with [Caddy][caddy].

-   OAuth2 (with hashed password and Bearer with JWT) based authentication

-   [CORS (Cross Origin Resource Sharing)][cors] enabled.

-   Flask inspired divisional folder structure for better decoupling and encapsulation. This
    is suitable for small to medium backend development.

-   Dockerized using **python:3.12-slim-bookworm** and optimized for size and functionality.
    Dockerfile for Python 3.11 and 3.10 can also be found in the `dockerfiles` directory.

## Quickstart

### Run the app in containers

-   Clone the repo and navigate to the root folder.

-   To run the app using Docker, make sure you've got [Docker][docker] installed on your
    system. From the project's root directory, run:

    ```sh
    docker compose up -d
    ```

### Or, run the app locally

If you want to run the app locally, without using Docker, then:

-   Clone the repo and navigate to the root folder.

-   Create a virtual environment. Here I'm using Python's built-in venv in a Unix system.
    Run:

    ```sh
    python3.12 -m venv .venv
    ```

-   Activate the environment. Run:

    ```sh
    source .venv/bin/activate
    ```

-   Go to the folder created by cookie-cutter (default is **fastapi-nano**).

-   Install the dependencies. Run:

    ```bash
    pip install -r requirements.txt -r requirements-dev.txt
    ```

-   Start the app. Run:

    ```bash
    uvicorn app.main:app --port 5002 --reload
    ```

### Check the APIs

-   To play around with the APIs, go to the following link on your browser:

    ```
    http://localhost:5002/docs
    ```


-   Also, notice the `curl` section in the above screen shot. You can directly use the
    highlighted curl command in your terminal. Make sure you've got `jq` installed in your
    system.

    ```sh
    curl -X GET "http://localhost:5002/api_a/22" \
            -H "accept: application/json" \
            -H "Authorization: Bearer $(curl -X POST "http://localhost:5002/token" \
                            -H "accept: application/x-www-form-urlencoded" \
                            -d "username=ubuntu&password=debian" | jq -r ".access_token")"
    ```

    This should show a response like this:

    ```json
    {
        "seed": 22,
        "random_first": 5,
        "random_second": 13
    }
    ```

-   To test the `GET` APIs with Python, you can use a http client library like
    [httpx][httpx]:

    ```python
    import httpx

    with httpx.Client() as client:

        # Collect the API token.
        r = client.post(
            "http://localhost:5002/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"username": "ubuntu", "password": "debian"},
        )
        token = r.json()["access_token"]

        # Use the token value to hit the API.
        r = client.get(
            "http://localhost:5002/api_a/22",
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        )
        print(r.json())
    ```

## Folder structure

This shows the folder structure of the default template.

```txt
fastapi-nano
├── app                           # primary app folder
│   ├── apis                      # this houses all the API packages
│   │   ├── api_a                 # api_a package
│   │   │   ├── __init__.py       # empty init file to make the api_a folder a package
│   │   │   ├── mainmod.py        # main module of api_a package
│   │   │   └── submod.py         # submodule of api_a package
│   │   └── api_b                 # api_b package
│   │       ├── __init__.py       # empty init file to make the api_b folder a package
│   │       ├── mainmod.py        # main module of api_b package
│   │       └── submod.py         # submodule of api_b package
│   ├── core                      # this is where the configs live
│   │   ├── auth.py               # authentication with OAuth2
│   │   ├── config.py             # sample config file
│   │   └── __init__.py           # empty init file to make the config folder a package
│   ├── __init__.py               # empty init file to make the app folder a package
│   ├── main.py                   # main file where the fastAPI() class is called
│   ├── routes                    # this is where all the routes live
│   │   └── views.py              # file containing the endpoints of api_a and api_b
│   └── tests                     # test package
│       ├── __init__.py           # empty init file to make the tests folder a package
│       ├── test_api.py           # integration testing the API responses
│       └── test_functions.py     # unit testing the underlying functions
├── dockerfiles                   # directory containing all the dockerfiles
├── .env                          # env file containing app variables
├── Caddyfile                     # simple reverse-proxy with caddy
├── docker-compose.yml            # docker-compose file
├── pyproject.toml                # pep-518 compliant config file
├── requrements-dev.in            # .in file to enlist the top-level dev requirements
├── requirements-dev.txt          # pinned dev dependencies
├── requirements.in               # .in file to enlist the top-level app dependencies
└── requirements.txt              # pinned app dependencies
```

<div align="center">
✨ 🍰 ✨
</div>
