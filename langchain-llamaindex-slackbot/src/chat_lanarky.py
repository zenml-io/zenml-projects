from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from lanarky import LangchainRouter
from lanarky.testing import mount_gradio_app
from langchain import HuggingFaceHub, OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from pydantic import BaseSettings
from zenml.logger import get_logger

from slackbot_utils import get_vector_store

logger = get_logger(__name__)

model = "openai"

if model == "openai":
    openai_llm = OpenAI(
        temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo"
    )
    llm = openai_llm
elif model == "huggingface":
    huggingface_llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature": 0, "max_length": 500},
    )
    llm = huggingface_llm
else:
    raise ValueError(f"Invalid model argument: {model}")

# connect_to_zenml_server()

vectorstore = get_vector_store()


class Settings(BaseSettings):
    """
    Settings class for this application.
    Utilizes the BaseSettings from pydantic for environment variables.
    """

    openai_api_key: str

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    """Function to get and cache settings.
    The settings are cached to avoid repeated disk I/O.
    """
    return Settings()


def create_chain():
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(
            temperature=0,
            streaming=True,
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )


app = mount_gradio_app(FastAPI(title="RetrievalQAWithSourcesChainDemo"))
templates = Jinja2Templates(directory="templates")
chain = create_chain()


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


langchain_router = LangchainRouter(
    langchain_url="/chat", langchain_object=chain, streaming_mode=1
)
langchain_router.add_langchain_api_route(
    "/chat_json", langchain_object=chain, streaming_mode=2
)
langchain_router.add_langchain_api_websocket_route(
    "/ws", langchain_object=chain
)

app.include_router(langchain_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
