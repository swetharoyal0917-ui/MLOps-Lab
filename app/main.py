from fastapi import FastAPI

app = FastAPI(
    title="Advanced CI Pipeline API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "CI Pipeline working successfully"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }