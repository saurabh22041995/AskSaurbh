import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("cv")  # Single PDF loader
        st.session_state.docs = st.session_state.loader.load()

        if not st.session_state.docs:
            st.error("No documents found!")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        if not st.session_state.final_documents:
            st.error("No documents to create embeddings from!")
            return

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        # st.write("Vector Database is ready")
    else:
        st.write("")

user_prompt = st.text_input("Enter your query for saurabh CV")
if st.button("Ask"):
    with st.spinner('Wait for it...'):
        create_vector_embedding()

    if user_prompt:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        try:
            response = retrieval_chain.invoke({'input': user_prompt})
            st.write(response['answer'])

            if 'context' in response:
                with st.expander("Detailed View"):
                    for i, doc in enumerate(response['context']):
                        st.write(doc.page_content)
                        st.write("----------------------")
            else:
                st.write("No similar documents found.")
        except Exception as e:
            st.error(f"Error during query processing: {e}")
