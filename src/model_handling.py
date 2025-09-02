from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.utils import load_config

load_dotenv()

config = load_config("config.yaml")

if config["configurable"]["model_provider_sg"] == "openai":
    model_sg = config["configurable"]["model_sg"]
    model_provider_sg = config["configurable"]["model_provider_sg"]
    max_retries_sg = config["configurable"].get("max_retries_sg", 25)
    temperature_sg = config["configurable"].get("temperature_sg", None)
    reasoning_effort_sg = config["configurable"].get("reasoning_effort_sg", None)
    llm_sg = ChatOpenAI(model=model_sg, max_retries=max_retries_sg, temperature=temperature_sg, reasoning_effort=reasoning_effort_sg, timeout=60)

elif config["configurable"]["model_provider_sg"] == "google_genai":
    model_sg = config["configurable"]["model_sg"]
    model_provider_sg = config["configurable"]["model_provider_sg"]
    temperature_sg = config["configurable"].get("temperature_sg", None)
    reasoning_effort_sg = config["configurable"].get("reasoning_effort_sg", None)
    llm_sg = ChatGoogleGenerativeAI(model=model_sg)

elif config["configurable"]["model_provider_sg"] == "groq":
    model_sg = config["configurable"]["model_sg"]
    model_provider_sg = config["configurable"]["model_provider_sg"]
    temperature_sg = config["configurable"].get("temperature_sg", None)
    reasoning_effort_sg = config["configurable"].get("reasoning_effort_sg", None)
    llm_sg = ChatGroq(model=model_sg)

elif config["configurable"]["model_provider_sg"] == "ollama":
    model_sg = config["configurable"]["model_sg"]
    model_provider_sg = config["configurable"]["model_provider_sg"]
    temperature_sg = config["configurable"].get("temperature_sg", None)
    reasoning_effort_sg = config["configurable"].get("reasoning_effort_sg", None)
    llm_sg = ChatOllama(model=model_sg, temperature=temperature_sg)


if config["configurable"]["model_provider_tg"] == "openai":
    model_tg = config["configurable"]["model_tg"]
    model_provider_tg = config["configurable"]["model_provider_tg"]
    max_retries_tg = config["configurable"].get("max_retries_tg", 25)
    temperature_tg = config["configurable"].get("temperature_tg", None)
    reasoning_effort_tg = config["configurable"].get("reasoning_effort_tg", None)
    llm_tg = ChatOpenAI(model=model_tg, max_retries=max_retries_tg, temperature=temperature_tg, reasoning_effort=reasoning_effort_tg, timeout=60)

elif config["configurable"]["model_provider_tg"] == "google_genai":
    model_tg = config["configurable"]["model_tg"]
    model_provider_tg = config["configurable"]["model_provider_tg"]
    max_retries_tg = config["configurable"].get("max_retries_tg", 25)
    temperature_tg = config["configurable"].get("temperature_tg", None)
    reasoning_effort_tg = config["configurable"].get("reasoning_effort_tg", None)
    llm_tg = ChatGoogleGenerativeAI(model=model_tg)

elif config["configurable"]["model_provider_tg"] == "groq":
    model_tg = config["configurable"]["model_tg"]
    model_provider_tg = config["configurable"]["model_provider_tg"]
    max_retries_tg = config["configurable"].get("max_retries_tg", 25)
    temperature_tg = config["configurable"].get("temperature_tg", None)
    reasoning_effort_tg = config["configurable"].get("reasoning_effort_tg", None)
    llm_tg = ChatGroq(model=model_tg)

elif config["configurable"]["model_provider_tg"] == "ollama":
    model_tg = config["configurable"]["model_tg"]
    model_provider_tg = config["configurable"]["model_provider_tg"]
    max_retries_tg = config["configurable"].get("max_retries_tg", 25)
    temperature_tg = config["configurable"].get("temperature_tg", None)
    reasoning_effort_tg = config["configurable"].get("reasoning_effort_tg", None)
    llm_tg = ChatOllama(model=model_tg)