# Mostly taken from the `chat-langchain` demo application
# https://github.com/hwchase17/chat-langchain

"""Main entrypoint for the web chat app."""
import logging
from typing import Dict, List, Optional

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore
from query_data import get_openai_chain
from schemas import ChatResponse
from zenml.enums import ExecutionStatus

app = FastAPI()
templates = Jinja2Templates(directory="templates")
versions: List[str] = []
versions_to_index: Dict[str, VectorStore] = {}
active_version: Optional[str] = None
vectorstore: Optional[VectorStore] = None


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    from zenml.post_execution import get_pipeline

    global versions
    global versions_to_index
    global vectorstore
    global active_version

    pipeline = get_pipeline("zenml_docs_index_generation")
    for run_ in pipeline.runs:
        if run_.status != ExecutionStatus.COMPLETED:
            continue
        version = run_.name.split("_")[-1]
        if len(version.split(".")) != 3:
            continue
        vectorstore = run_.steps[-1].output.read()
        versions_to_index[version] = vectorstore
        versions.append(version)

    if not versions:
        raise Exception(
            "No index versions found. Please run `python run.py` first."
        )

    versions = sorted(
        versions, key=lambda x: [int(y) for y in x.split(".")], reverse=True
    )
    print(f"Found versions: {versions}")
    active_version = versions[0]
    vectorstore = versions_to_index[versions[0]]


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/versions", response_model=List[str])
async def get_versions():
    global versions
    return versions


@app.get("/versions/active", response_model=str)
async def get_active_version():
    global active_version
    return active_version


@app.post("/versions/{version}", response_model=str)
async def set_active_version(version: str):
    global versions_to_index
    global vectorstore
    global active_version
    if not version in versions_to_index:
        raise Exception(f"Version '{version}' not found.")
    vectorstore = versions_to_index[version]
    active_version = version
    return "success"


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_openai_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8242)
