from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI


def llamaModel(**chain_kwargs):
    llama_llm = LlamaCpp(**chain_kwargs, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    return llama_llm


def gpt_model(**chain_kwargs):
    gpt_llm = ChatOpenAI(**chain_kwargs)
    return gpt_llm


MODEL = {"llama": llamaModel, "gpt": gpt_model}


class Model:
    def __init__(self, model_instance):
        self.model_instance = model_instance

    @classmethod
    def from_config(cls, type: str, name: str, **kwargs):
        try:
            model_instance = MODEL[type](**kwargs)
        except KeyError:
            raise KeyError(f'Mode type "{type}" & "{name}" is not implemented.')

        return cls(model_instance)
