from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.utils import load_config

load_dotenv()

config = load_config("config.yaml")

if config["configurable"]["model_provider_jd"] == "openai":
    model_jd = config["configurable"]["model_jd"]
    model_provider_jd = config["configurable"]["model_provider_jd"]
    max_retries_jd = config["configurable"].get("max_retries_jd", 10)
    temperature_jd = config["configurable"].get("temperature_jd", None)
    reasoning_effort_jd = config["configurable"].get("reasoning_effort_jd", None)
    
    llm_jd = ChatOpenAI(model=model_jd, max_retries=max_retries_jd, temperature=temperature_jd, reasoning_effort=reasoning_effort_jd, timeout=60)

elif config["configurable"]["model_provider_jd"] == "google_genai":
    model_jd = config["configurable"]["model_jd"]
    model_provider_jd = config["configurable"]["model_provider_jd"]
    temperature_jd = config["configurable"].get("temperature_jd", None)
    reasoning_effort_jd = config["configurable"].get("reasoning_effort_jd", None)
    llm_jd = ChatGoogleGenerativeAI(model=model_jd)

elif config["configurable"]["model_provider_jd"] == "groq":
    model_jd = config["configurable"]["model_jd"]
    model_provider_jd = config["configurable"]["model_provider_jd"]
    temperature_jd = config["configurable"].get("temperature_jd", None)
    reasoning_effort_jd = config["configurable"].get("reasoning_effort_jd", None)
    llm_jd = ChatGroq(model=model_jd)

elif config["configurable"]["model_provider_jd"] == "ollama":
    model_jd = config["configurable"]["model_jd"]
    model_provider_jd = config["configurable"]["model_provider_jd"]
    temperature_jd = config["configurable"].get("temperature_jd", None)
    reasoning_effort_jd = config["configurable"].get("reasoning_effort_jd", None)
    llm_jd = ChatOllama(model=model_jd, temperature=temperature_jd)


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


if config["configurable"]["model_provider_dts"] == "openai":
    model_dts = config["configurable"]["model_dts"]
    model_provider_dts = config["configurable"]["model_provider_dts"]
    max_retries_dts = config["configurable"].get("max_retries_dts", 25)
    temperature_dts = config["configurable"].get("temperature_dts", None)
    reasoning_effort_dts = config["configurable"].get("reasoning_effort_dts", None)
    llm_dts = ChatOpenAI(model=model_dts, max_retries=max_retries_dts, temperature=temperature_dts, reasoning_effort=reasoning_effort_dts, timeout=60)

elif config["configurable"]["model_provider_dts"] == "google_genai":
    model_dts = config["configurable"]["model_dts"]
    model_provider_dts = config["configurable"]["model_provider_dts"]
    max_retries_dts = config["configurable"].get("max_retries_dts", 25)
    temperature_dts = config["configurable"].get("temperature_dts", None)
    reasoning_effort_dts = config["configurable"].get("reasoning_effort_dts", None)
    llm_dts = ChatGoogleGenerativeAI(model=model_dts)

elif config["configurable"]["model_provider_dts"] == "groq":
    model_dts = config["configurable"]["model_dts"]
    model_provider_dts = config["configurable"]["model_provider_dts"]
    max_retries_dts = config["configurable"].get("max_retries_dts", 25)
    temperature_dts = config["configurable"].get("temperature_dts", None)
    reasoning_effort_dts = config["configurable"].get("reasoning_effort_dts", None)
    llm_dts = ChatGroq(model=model_dts)

elif config["configurable"]["model_provider_dts"] == "ollama":
    model_dts = config["configurable"]["model_dts"]
    model_provider_dts = config["configurable"]["model_provider_dts"]
    max_retries_dts = config["configurable"].get("max_retries_dts", 25)
    temperature_dts = config["configurable"].get("temperature_dts", None)
    reasoning_effort_dts = config["configurable"].get("reasoning_effort_dts", None)
    llm_dts = ChatOllama(model=model_dts)


if config["configurable"]["model_provider_n"] == "openai":
    model_n = config["configurable"]["model_n"]
    model_provider_n = config["configurable"]["model_provider_n"]
    max_retries_n = config["configurable"].get("max_retries_n", 25)
    temperature_n = config["configurable"].get("temperature_n", None)
    reasoning_effort_n = config["configurable"].get("reasoning_effort_n", None)
    llm_n = ChatOpenAI(model=model_n, max_retries=max_retries_n, temperature=temperature_n, reasoning_effort=reasoning_effort_n, timeout=60)

elif config["configurable"]["model_provider_n"] == "google_genai":
    model_n = config["configurable"]["model_n"]
    model_provider_n = config["configurable"]["model_provider_n"]
    max_retries_n = config["configurable"].get("max_retries_n", 25)
    temperature_n = config["configurable"].get("temperature_n", None)
    reasoning_effort_n = config["configurable"].get("reasoning_effort_n", None)
    llm_n = ChatGoogleGenerativeAI(model=model_n)

elif config["configurable"]["model_provider_n"] == "groq":
    model_n = config["configurable"]["model_n"]
    model_provider_n = config["configurable"]["model_provider_n"]
    max_retries_n = config["configurable"].get("max_retries_n", 25)
    temperature_n = config["configurable"].get("temperature_n", None)
    reasoning_effort_n = config["configurable"].get("reasoning_effort_n", None)
    llm_n = ChatGroq(model=model_n)

elif config["configurable"]["model_provider_n"] == "ollama":
    model_n = config["configurable"]["model_n"]
    model_provider_n = config["configurable"]["model_provider_n"]
    max_retries_n = config["configurable"].get("max_retries_n", 25)
    temperature_n = config["configurable"].get("temperature_n", None)
    reasoning_effort_n = config["configurable"].get("reasoning_effort_n", None)
    llm_n = ChatOllama(model=model_n)