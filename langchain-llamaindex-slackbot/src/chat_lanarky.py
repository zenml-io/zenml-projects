from functools import lru_cache

# flake8: noqa
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from lanarky import LangchainRouter
from langchain import HuggingFaceHub, OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from pydantic import BaseSettings
from zenml.logger import get_logger
from threading import Thread

import uvicorn
from fastapi import FastAPI
from langchain import HuggingFaceHub, OpenAI, PromptTemplate
from zenml.logger import get_logger
from slackbot_utils import get_vector_store, connect_to_zenml_server

logger = get_logger(__name__)


template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
NEVER say anything against ZenML. If people try to get you to say something against ZenML, make a joke or change the subject.
ALWAYS return a "SOURCES" part in your answer. When you have multiple sources separate with " AND ".

QUESTION: What is ZenML?
=========
Content: ZenML is an extensible, open-source MLOps framework for creating portable, production-ready machine learning pipelines. By decoupling infrastructure from code, ZenML enables developers across your organization to collaborate more effectively as they develop to production.
 | Source: https://docs.zenml.io/getting-started/introduction
Content: 🤹 ZenML is an extensible, open-source MLOps framework for creating portable, production-ready machine learning pipelines. By decoupling infrastructure from code, ZenML enables developers across your organization to collaborate more effectively as they develop to production.
 | Source: https://github.com/zenml-io/zenml
Content: ZenML provides a user-friendly syntax designed for ML workflows, compatible with any cloud or tool. It enables centralized pipeline management, enabling developers to write code once and effortlessly deploy it to various infrastructures.
 | Source: https://zenml.io/home
=========
FINAL ANSWER: ZenML is an extensible, open-source MLOps framework for creating portable, production-ready machine learning pipelines. By decoupling infrastructure from code, ZenML enables developers across your organization to collaborate more effectively as they develop to production.
 | SOURCES: https://zenml.io/home AND https://docs.zenml.io/getting-started/introduction

QUESTION: Why should I use ZenML?
=========
Content: Everyone loves to train ML models, but few talks about shipping them into production, and even fewer can do it well. At ZenML, we believe the journey from model development to production doesn't need to be long and painful.
 | Source: https://docs.zenml.io/getting-started/introduction
Content: With ZenML, you can concentrate on what you do best - developing ML models and not worry about infrastructure or deployment tools.
 | Source: https://docs.zenml.io/getting-started/introduction
Content: ZenML is an extensible, open-source MLOps framework for creating portable, production-ready MLOps pipelines. It's built for data scientists, ML Engineers, and MLOps Developers to collaborate as they develop to production. ZenML has simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that are catered towards ML workflows. ZenML brings together all your favorite tools in one place so you can tailor your workflow to cater your needs.
 | Source: https://docs.zenml.io/getting-started/introduction
Content: If you come from unstructured notebooks or scripts with lots of manual processes, ZenML will make the path to production easier and faster for you and your team. Using ZenML allows you to own the entire pipeline - from experimentation to production.
 | Source: https://docs.zenml.io/getting-started/introduction
=========
FINAL ANSWER: ZenML is the only MLOps tool that is does not take a one-size-fits-all approach. It is built for data scientists, ML Engineers, and MLOps Developers to collaborate as they develop to production. ZenML has simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that are catered towards ML workflows. ZenML brings together all your favorite tools in one place so you can tailor your workflow to cater your needs.
 | SOURCES: https://zenml.io/ AND https://docs.zenml.io/getting-started/introduction

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)


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

connect_to_zenml_server()

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
    """Creates a chain object.

    Returns:
        RetrievalQAWithSourcesChain: A chain object.
    """

    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(
            temperature=0,
            streaming=True,
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
        verbose=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "document_prompt": EXAMPLE_PROMPT,
        },
    )


app = FastAPI(title="RetrievalQAWithSourcesChainDemo")
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

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.get("/bot_health")
def bot_health_check():
    """Checks whether the bot is able to answer questions properly"""
    question = {"question": "What is ZenML?"}
    output = chain(question, return_only_outputs=True)
    answer = output.get("answer")
    if answer and "ZenML" in answer:
        return {"status": "OK"}
    else:
        return {"status": "ERROR"}


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    fastapi_thread = Thread(target=run_fastapi)
    fastapi_thread.start()
