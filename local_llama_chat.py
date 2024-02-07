from llama_cpp import Llama
import streamlit as st
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

MODEL_NAME = 'llama-2-7b-chat.Q4_K_M.gguf'
MODEL_PATH = "model path"
# Number of threads to use
NUM_THREADS = 8

class CustomLLM(LLM):
    model_name = MODEL_NAME

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        p = f"Human: {prompt} Assistant: "
        prompt_length = len(p)
        llm = Llama(model_path=MODEL_PATH, n_threads=NUM_THREADS)
        output = llm(p, max_tokens=512, stop=["Human:"], echo=True)['choices'][0]['text']
        # only return newly generated tokens by slicing list to include words after the original prompt
        response = output[prompt_length:]
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


def clear_convo():
    st.session_state['messages'] = []


def init():
    st.set_page_config(page_title='Local Llama', page_icon=':robot_face: ')
    st.sidebar.title('Local Llama')
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []


if __name__ == '__main__':
    init()


    @st.cache_resource
    def get_llm():
        llm = CustomLLM()
        return llm

    clear_button = st.sidebar.button("Clear Conversation", key="clear", on_click=clear_convo)

    user_input = st.chat_input("Say something")

    if user_input:
        llm = get_llm()
        llm._call(prompt=user_input)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])