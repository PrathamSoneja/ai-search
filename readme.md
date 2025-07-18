### Create and activate virtual environment
    python -m venv .venv

    .venv\Scripts\activate

### Install all the required packages
    pip install -r requirements.txt

### Run the fastapi application
    uvicorn app:app --host 127.0.0.1 --port 8000

### Documentation
    http://127.0.0.1:8000/docs

    search api -> `http://127.0.0.1:8000/search?query=${encodeURIComponent(query)}&top_k=5`
        query is a string argument
        top_k is an integer argument
    
    upload api -> `http://127.0.0.1:8000/upload`
    