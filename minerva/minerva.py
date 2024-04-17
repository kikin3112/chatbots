from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Import CSS styling and chat message templates from html.py
#from custom_html import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

#def get_vectors(text_chunks):
    embedded = OpenAIEmbeddings()
    vectors = FAISS.from_texts(texts=text_chunks, embedding=embedded)
    return vectors

def get_convo_chain(vectors):
    llm = ChatOpenAI()
    memoria = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectors.as_retriever(),
        memory=memoria
    )

def handle_question(pregunta):
    respuesta = st.session_state.conversacion({'pregunta': pregunta})
    st.write(respuesta)

def main():
    load_dotenv()
    st.set_page_config(page_icon=':brain:', page_title='Minerva')

    #st.write(css, unsafe_allow_html=True)  # Apply CSS styling

    if 'conversacion' not in st.session_state:
        st.session_state.conversacion = None

    st.header('Pregúntame algo sobre tus PDFs')
    pregunta = st.text_input('Hazme una pregunta:')
    if pregunta:
        handle_question(pregunta)

    #st.write(bot_template.replace('{{MSG}}', 'Hola humano. Soy Minerva.'), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader('Tus documentos')
        pdf_docs = st.file_uploader("Sube aquí tus PDFs y haz click en 'Procesar'",
                         type='pdf',
                         accept_multiple_files=True)
        if st.button('Procesar'):
            with st.spinner('Estudiando...'):
                # get text
                raw_text = get_pdf_text(pdf_docs)
                
                # get chunks
                text_chunks = get_text_chunks(raw_text)
                
                # store vectors
                #vectors = get_vectors(text_chunks)

                # conversaciones
                #st.session_state.conversacion = get_convo_chain(vectors)

if __name__ == "__main__":
    main()
