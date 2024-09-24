# tools
import os
import glob
from io import BytesIO
from dotenv import load_dotenv
from tempfile import TemporaryDirectory

# llamaindex
import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, SimpleDirectoryReader, ServiceContext, VectorStoreIndex, Settings

# streamlit
import streamlit as st

# keys
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


def response_generator(stream):
    """
    Generator that yields chunks of data from a stream response.
    
    Args:
        stream: The stream object from which to read data chunks.
    Yields:
        bytes: The next chunk of data from the stream response.
    """
    for chunk in stream.response_gen:
        yield chunk

@st.cache_resource(show_spinner=False)
def load_data(documents: list[BytesIO]) -> VectorStoreIndex:
    """
    Loads and indexes multiple PDF documents using Ollama and Llamaindex.
    This function takes a list of documents as input and performs the following actions:

    1. Initializes the LLM model.
    2. Writes the uploaded PDFs to a temporary directory.
    3. Loads and processes each document using SimpleDirectoryReader.
    4. Splits the text into sentences and generates embeddings.
    5. Creates a VectorStoreIndex from the processed documents.

    Args:
        documents (list[BytesIO]): List of documents to query.

    Returns:
        VectorStoreIndex: An instance of VectorStoreIndex containing the indexed documents and embeddings.
    """

    # Initialize LLM
    llm = OpenAI(model="gpt-4o-mini")

    try:
        with TemporaryDirectory() as tmpdir:
            # Save all uploaded PDFs to the temporary directory
            for idx, document in enumerate(documents, start=1):
                temp_file_path = os.path.join(tmpdir, f'temp_{idx}.pdf')
                with open(temp_file_path, 'wb') as f:
                    f.write(document.getbuffer())

            with st.spinner(text="Loading and indexing the Streamlit docs. This may take a few minutes."):
                # Loading documents
                docs = SimpleDirectoryReader(tmpdir).load_data()

                # Embeddings | Query Container
                text_splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=150)
                embed_model = OpenAIEmbedding(model_name="text-embedding-3-large", embed_batch_size=200)

                # Settings
                Settings.llm = llm
                Settings.embed_model = embed_model
                Settings.text_splitter = text_splitter
                Settings.transformations = [text_splitter]

                system_prompt = (
                    "Sen, işe alım konularında uzman, Türkçe konuşan ve "
                    "görevlerini iyi bilen bir işe alım asistanısın. "
                    "İnsan adaylarına CV'lerini analiz ederek geri bildirimler sağlıyorsun "
                    "ve işe alım sürecini destekliyorsun."
                )

                # Indexing DB
                index = VectorStoreIndex.from_documents(
                    docs,
                    embed_model=embed_model,
                    transformations=Settings.transformations
                )
    except Exception as e:
        st.error(f"An error occurred while processing the files: {e}")
        return None

    return index

def main() -> None:
    """
    Controls the main chat application logic using Streamlit and Ollama.
    This function serves as the primary orchestrator of a chat application with the following tasks:

    1. Page Configuration: Sets up the Streamlit page's title, icon, layout, and sidebar using st.set_page_config.
    2. Model Selection: Manages model selection using st.selectbox and stores the chosen model in Streamlit's session state.
    3. Chat History Initialization: Initializes the chat history list in session state if it doesn't exist.
    4. Data Loading and Indexing: Calls the load_data function to create a VectorStoreIndex from the provided model name.
    5. Chat Engine Initialization: Initializes the chat engine using the VectorStoreIndex instance, enabling context-aware and streaming responses.
    6. Chat History Display: Iterates through the chat history messages and presents them using Streamlit's chat message components.
    7. User Input Handling:
          - Accepts user input through st.chat_input.
          - Appends the user's input to the chat history.
          - Displays the user's message in the chat interface.
    8. Chat Assistant Response Generation:
          - Uses the chat engine to generate a response to the user's prompt.
          - Displays the assistant's response in the chat interface, employing st.write_stream for streaming responses.
          - Appends the assistant's response to the chat history.
    """

    st.set_page_config(page_title="Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with documents 💬")
    
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    with st.sidebar:
        # LLM
        llm = OpenAI(model="gpt-4o-mini")

        # Data ingestion
        documents = st.file_uploader("Upload PDF files to query", type=['pdf'], accept_multiple_files=True)

        # File processing                
        if st.button('Process files'):
            if documents:
                index = load_data(documents)
                if index is not None:
                    st.session_state.activate_chat = True
                else:
                    st.session_state.activate_chat = False
            else:
                st.warning("Please upload at least one PDF file before processing.")

    if st.session_state.activate_chat:
        # Initialize chat history                   
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize the chat engine if not already done
        if "chat_engine" not in st.session_state and index is not None:
            st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", streaming=True)

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("How can I help you?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Chat assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                message_placeholder = st.empty()
                with st.chat_message("assistant"):
                    stream = st.session_state.chat_engine.stream_chat(prompt)
                    response = st.write_stream(response_generator(stream))
                st.session_state.messages.append({"role": "assistant", "content": response})
                
    else:
        st.markdown(
            "<span style='font-size:15px;'><b>Upload PDF files to start chatting</b></span>",
            unsafe_allow_html=True
        )

if __name__=='__main__':
    main()
