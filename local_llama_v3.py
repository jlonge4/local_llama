import os
from pathlib import Path

import streamlit as st
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from ollama import generate

os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"


def get_doc_store():
    return ChromaDocumentStore(
        collection_name="mydocs", persist_path="./vec-index", distance_function="cosine"
    )


def get_context(query):
    document_store = get_doc_store()

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", OllamaTextEmbedder())
    query_pipeline.add_component(
        "retriever", ChromaEmbeddingRetriever(document_store=document_store, top_k=3)
    )

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    result = query_pipeline.run({"text_embedder": {"text": query}})
    context = [doc.content for doc in result["retriever"]["documents"]]
    sources = [doc.meta["page_number"] for doc in result["retriever"]["documents"]]
    files = [doc.meta["file_path"] for doc in result["retriever"]["documents"]]
    final_context = [
        f"Context: {c} (Page: {s}, File: {f})"
        for c, s, f in zip(context, sources, files)
    ]
    # Uncomment for debug st.write(final_context)
    return final_context


def indexing_pipe(filename):
    document_store = get_doc_store()

    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component(
        "cleaner",
        DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=True,
        ),
    )
    pipeline.add_component(
        "splitter",
        DocumentSplitter(split_by="word", split_length=300, split_overlap=15),
    )
    pipeline.add_component("embedder", OllamaDocumentEmbedder())
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder.documents", "writer")

    os.makedirs("uploads", exist_ok=True)
    # Save the file to disk
    file_path = os.path.join("uploads", filename.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    pipeline.run({"converter": {"sources": [Path(file_path)]}})


def invoke_ollama(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    system = f"""You are a helpful assistant that answers users questions and chats. 
        History has been provided in the <history> tags. You must not mention your knowledge of the history to the user,
        only use it to answer follow up questions if needed.
        {{history}}
        {st.session_state.messages}
        {{history}}

        Context to help you answer user's questions have been provided in the <context> tags.
        {{context}}
        {get_context(user_input)}
        {{context}}
        Use ONLY the history and or context provided to answer the question.
        Use as few words as possible to accurately answer."""
    # Uncomment to make llama use a template {"answer": "the answer"}
    # Use the following template: {json.dumps(template)}."""
    prompt_wrapper = f"""You are a helpful assistant that answers users questions and chats. 
    Use the provided history and context to answer the question. 
    {{user_query}}
    {user_input}
    {{user_query}}
    Use as few words as possible to accurately answer, providing citations to the page number and file path from which your answer was synthesized."""

    data = {
        "prompt": prompt_wrapper,
        "model": "llama3:8b",
        "format": "json",
        "stream": True,
        "options": {"top_p": 0.05, "top_k": 5},
    }
    s = ""
    box = st.chat_message("assistant").empty()

    for part in generate(
        model=data["model"],
        prompt=data["prompt"],
        system=system,
        # Format seems equivelant to enforcing a template within the prompt
        # format=data["format"],
        options=data["options"],
        stream=data["stream"],
    ):
        s += part["response"]
        box.write(s)

    st.session_state.messages.append({"role": "assistant", "content": s})


def clear_convo():
    st.session_state["messages"] = []


def init():
    st.set_page_config(page_title="Local Llama", page_icon=":robot_face: ")
    st.sidebar.title("Local Llama")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


if __name__ == "__main__":
    init()

    clear_button = st.sidebar.button(
        "Clear Conversation", key="clear", on_click=clear_convo
    )
    file = st.file_uploader(
        "Choose a file to index...", type=["docx", "pdf", "txt", "md"]
    )

    # display on sidebar all files within uploads dir
    st.sidebar.markdown("## Uploaded Files")
    uploaded_files = os.listdir("uploads")
    for f in uploaded_files:
        st.sidebar.markdown(f)
    st.sidebar.info(
        """This application stores uploaded files in the 'uploads' directory upon upload and then indexes them into a 
                    locally persisted Chroma Document Store so that you may re-use your documentation as necessary."""
    )
    clicked = st.button("Upload File", key="Upload")
    if file and clicked:
        with st.spinner("Wait for it..."):
            indexing_pipe(file)
        st.success("Indexed {0}!".format(file.name))
    user_input = st.chat_input("Say something")

    if user_input:
        invoke_ollama(user_input=user_input)
