# semantic_search_api
python -m pip install --upgrade pip
pip install farm-haystack[inference]
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall
pip install --upgrade pymupdf
pip install thefuzz


pip install fastapi
pip install "uvicorn[standard]"

#launch:
uvicorn --host 0.0.0.0 main:app