### Create and activate virtual environment
    python -m venv .venv

    .venv\Scripts\activate

### Install all the required packages
    pip install -r requirements.txt

### Run the fastapi application
    uvicorn app:app --host 127.0.0.1 --port 8000
    