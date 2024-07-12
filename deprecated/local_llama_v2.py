from llama_cpp import Llama
import streamlit as st
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from haystack.pipelines import Pipeline
from haystack.pipelines.standard_pipelines import DocumentSearchPipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, TextConverter, FileTypeClassifier, PDFToTextConverter, MarkdownConverter, DocxToTextConverter, PreProcessor
import os

MODEL_NAME = 'llama-2-7b-chat.Q4_K_M.gguf'
MODEL_PATH = "Model Path"
# Number of threads to use
NUM_THREADS = 8

def get_doc_store():
    try:
        document_store = FAISSDocumentStore.load(index_path="my_index.faiss", config_path="my_config.json")
    except:
        document_store = FAISSDocumentStore(embedding_dim=768)
        document_store.save(index_path="my_index.faiss", config_path="my_config.json")
    return document_store

def get_context(query):
    document_store = get_doc_store()
    retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/msmarco-bert-base-dot-v5",
            model_format="sentence_transformers"
    )
    pipe = DocumentSearchPipeline(retriever)
    top_k = 1
    answer = pipe.run(
                query=query,
                params={
                    "Retriever": {
                        "top_k": top_k,
                    },
                }
            )
    # st.write(answer['documents'])
    return answer['documents']


def indexing_pipe(filename):
    document_store = get_doc_store()
    file_type_classifier = FileTypeClassifier()

    text_converter = TextConverter()
    pdf_converter = PDFToTextConverter()
    md_converter = MarkdownConverter()
    docx_converter = DocxToTextConverter()
    preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_by="word",
            split_length=300,
            split_overlap=20,
            split_respect_sentence_boundary=True,
        )
    
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/msmarco-bert-base-dot-v5",
        model_format="sentence_transformers"
    )

    # indexing pipeline
    p = Pipeline()
    p.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["File"])
    p.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
    p.add_node(component=pdf_converter, name="PdfConverter", inputs=["FileTypeClassifier.output_2"])
    p.add_node(component=md_converter, name="MarkdownConverter", inputs=["FileTypeClassifier.output_3"])
    p.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_4"])
    p.add_node(
        component=preprocessor,
        name="Preprocessor",
        inputs=["TextConverter", "PdfConverter", "MarkdownConverter", "DocxConverter"],
    )
    p.add_node(component=retriever,name='Retriever', inputs=['Preprocessor'])
    p.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])

    os.makedirs("uploads", exist_ok=True)
    # Save the file to disk
    file_path = os.path.join("uploads", filename.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    # Run pipeline on document and add metadata to include doc name
    p.run(file_paths=['uploads/{0}'.format(filename.name)], meta={"document_name": filename.name})

    # Once documents are ran through the pipeline, use this to add embeddings to the datastore
    document_store.save(index_path="my_index.faiss", config_path="my_config.json")
    print(f'Docs match embedding count: {document_store.get_document_count() == document_store.get_embedding_count()}')


class CustomLLM(LLM):
    model_name = MODEL_NAME

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        context = get_context(prompt)
        p = f'Based on the context \n{context} \nAnswer the question {prompt}'
        prompt_length = len(p)
        llm = Llama(model_path=MODEL_PATH, n_threads=NUM_THREADS, n_ctx=2048,)
        output = llm(p, max_tokens=4016, stop=["Human:"], echo=True)['choices'][0]['text']
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
    file = st.file_uploader("Choose a file to index...", type=['docx', 'pdf', 'txt'])
    clicked = st.button('Upload File', key='Upload')
    if file and clicked:
        with st.spinner('Wait for it...'):
            document_store = indexing_pipe(file)
        st.success('Indexed {0}! Refresh to update indexes.'.format(file.name))

    user_input = st.chat_input("Say something")

    if user_input:
        llm = get_llm()
        llm._call(prompt=user_input)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])