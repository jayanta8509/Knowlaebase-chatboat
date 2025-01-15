import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import tempfile
import os

def main():
    st.title("PDF Knowledge Base Chatbot")
    
    # Set up OpenAI API key
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file is not None:
        # Process the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        documents = load_pdf(tmp_file_path)
        vectorstore = process_documents(documents)
        
        # Create the conversational chain
        chain = create_conversational_chain(vectorstore)
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your PDF"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = chain({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
                st.markdown(response['answer'])
            
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
    else:
        st.write("Please upload a PDF file to begin.")

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def process_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(texts, embeddings)

def create_conversational_chain(vectorstore):
    llm = OpenAI(temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

if __name__ == "__main__":
    main()