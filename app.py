import streamlit as st
import os
from dotenv import load_dotenv
# from langchain_community.embeddings import HuggingFaceEmbeddings # Old
from langchain_huggingface import HuggingFaceEmbeddings  # New
from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq # Remove or comment out
from langchain_google_genai import ChatGoogleGenerativeAI  # Import Google Gemini
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai  # For configuring the API key

# Suppress HuggingFace Tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

DB_FAISS_PATH = "faiss_index/"

# Custom prompt template (should work with Gemini, might need slight tweaks for optimal performance)
custom_template = """This is a conversation with an AI assistant.
The AI is helpful and provides comprehensive answers based ONLY on the context provided.
If the AI does not know the answer to a question, it truthfully says "I Don't know".
The AI will not make up information or answer questions outside of the provided context.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Helpful Answer:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)


@st.cache_resource  # Cache the embeddings model
def load_embeddings_model():
    print("Loading embeddings model...")
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                 model_kwargs={'device': 'cpu'})


@st.cache_resource  # Cache the FAISS index
def load_faiss_index(_embeddings):
    print("Loading FAISS index...")
    if not os.path.exists(DB_FAISS_PATH) or not os.listdir(DB_FAISS_PATH):
        st.error("FAISS index not found. Please run `create_embeddings.py` first.")
        return None
    return FAISS.load_local(DB_FAISS_PATH, _embeddings, allow_dangerous_deserialization=True)


@st.cache_resource  # Cache the LLM
def load_llm():
    print("Loading LLM (Google Gemini)...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        return None

    # Configure the Gemini API key globally for the genai library
    try:
        genai.configure(api_key=google_api_key)
    except Exception as e:
        st.error(f"Failed to configure Google Gemini API key: {e}")
        return None

    # model_name: "gemini-pro" is a common choice.
    # "gemini-1.5-flash-latest" or "gemini-1.5-pro-latest" are newer options.
    # Check Google AI Studio for available models.
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, convert_system_message_to_human=True)


def get_conversational_chain(vector_store, llm):
    if vector_store is None or llm is None:
        return None

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT},
        return_source_documents=True
    )
    return chain


def main():
    st.set_page_config(page_title="Customer Support Chatbot", layout="wide")
    st.title("📄 Customer Support Chatbot")
    st.markdown("Ask questions about your PSM Health Plan options.")

    embeddings = load_embeddings_model()
    db = load_faiss_index(embeddings)
    llm = load_llm()

    if db is None or llm is None:
        st.warning("System is not ready. Please check configurations and API keys.")
        st.stop()

    chain = get_conversational_chain(db, llm)
    if chain is None:
        st.error("Failed to initialize the conversational chain.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history_langchain" not in st.session_state:
        st.session_state.chat_history_langchain = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your plan..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("thinking..."):
            try:
                result = chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history_langchain})
                answer = result["answer"]

                st.session_state.chat_history_langchain.append((prompt, answer))
                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.chat_message("assistant"):
                    st.markdown(answer)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
                with st.chat_message("assistant"):
                    st.markdown(f"Sorry, an error occurred: {e}")


if __name__ == "__main__":
    main()
