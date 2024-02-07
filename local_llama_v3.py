import streamlit as st
from haystack.pipelines import Pipeline
from haystack.pipelines.standard_pipelines import DocumentSearchPipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, TextConverter, FileTypeClassifier, PDFToTextConverter, MarkdownConverter, DocxToTextConverter, PreProcessor
import os
import asyncio
from ollama import AsyncClient


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
    top_k = 3
    answer = pipe.run(
                query=query,
                params={
                    "Retriever": {
                        "top_k": top_k,
                    },
                }
            )
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
            split_length=350,
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


def invoke_ollama(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    async def chat():
        messages = [{
            'role': 'system',
            'content': f'''You are a helpful assistant that answers users questions and chats. 
            History has been provided in the <history> tags. You must not mention your knowledge of the history to the user,
            only use it to answer follow up questions if needed.
            {{history}}
            {st.session_state.messages}
            {{history}}

            Context to help you answer user's questions have been provided in the <context> tags.
            {{context}}
            {get_context(user_input)}
            {{context}}
            "role": "user",
            "content": {user_input}. Be concise''',
        },]
        s =''
        box = st.chat_message("assistant").empty()
        async for part in await AsyncClient(host='http://localhost:11434').chat(model='llama2', messages=messages, stream=True):
            s+=part['message']['content']
            box.write(s)
            
        st.session_state.messages.append({"role": "assistant", "content": s})

    asyncio.run(chat())

    # WITHOUT STREAMING
    # client = Client(host='http://localhost:11434')
    # response = client.chat(model='llama2', messages=[
    #     {
    #         'role': 'system',
    #         'content': f'''You are a helpful assistant that answers users questions and chats. 
    #         History has been provided in the <history> tags. You must not disclose your knowledge of the history to the user,
    #         only use it to answer follow up questions if needed.
    #         {{history}}
    #         {st.session_state.messages}
    #         {{history}}

    #         Context to help you answer user's questions have been provided in the <context> tags.
    #         {{context}}
    #         {get_context(user_input)}
    #         {{context}}
    #         "role": "user",
    #         "content": {user_input}''',
    #     },
    # ])
    # st.session_state.messages.append({"role": "user", "content": user_input})
    # st.session_state.messages.append({"role": "assistant", "content": response['message']['content']})


def clear_convo():
    st.session_state['messages'] = []


def init():
    st.set_page_config(page_title='Local Llama', page_icon=':robot_face: ')
    st.sidebar.title('Local Llama')
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []


if __name__ == '__main__':
    init()


    clear_button = st.sidebar.button("Clear Conversation", key="clear", on_click=clear_convo)
    file = st.file_uploader("Choose a file to index...", type=['docx', 'pdf', 'txt'])
    clicked = st.button('Upload File', key='Upload')
    if file and clicked:
        with st.spinner('Wait for it...'):
            document_store = indexing_pipe(file)
        st.success('Indexed {0}! Refresh to update indexes.'.format(file.name))
    user_input = st.chat_input("Say something")

    
    if user_input:
        invoke_ollama(user_input=user_input)

