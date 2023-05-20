from llama_cpp import Llama
from llama_index import download_loader, SimpleDirectoryReader, ServiceContext, LLMPredictor, GPTVectorStoreIndex, \
    PromptHelper, StorageContext, load_index_from_storage
from pathlib import Path
import os
import streamlit as st
from streamlit_chat import message
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
from typing import Optional, List, Mapping, Any
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext


MODEL_NAME = 'GPT4All-13B-snoozy.ggml.q4_0.bin'
MODEL_PATH = 'path_to_model'
#Number of threads to use 
NUM_THREADS = 8
# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

class CustomLLM(LLM):
    model_name = MODEL_NAME
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt) + 5
        llm = Llama(model_path=MODEL_PATH, n_threads=NUM_THREADS)

        output = llm(f"Q: {prompt} A: ", max_tokens=256,
                     stop=['Q: '], echo=True)['choices'][0]['text'].replace('A: ', '').strip()
        # only return newly generated tokens
        st.session_state.past.append(prompt)
        st.session_state['generated'].append(output)
        return output[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


# define our LLM
llm_predictor = LLMPredictor(llm=CustomLLM())
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper,
                                               embed_model=embed_model)\

def clear_convo():
    st.session_state['past'] = []
    st.session_state['generated'] = []


def init():
    st.set_page_config(page_title='Local ChatBot', page_icon=':robot_face: ')
    st.sidebar.title('Local ChatBot')


if __name__ == '__main__':
    init()

    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button:
        clear_convo()

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=75)
        submit_button = st.form_submit_button(label="Submit")

    if user_input and submit_button:
        llm = CustomLLM()
        llm._call(prompt=user_input)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state['generated'][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + "user")