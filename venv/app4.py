import pickle
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
google_api_key=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# palm.configure(api_key=google_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

st.title("Gemma Model Document Q&A")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Gemma-7b-It",
             temperature=0.3,top_p=0.95)

prompt_template = """
    Answer the questions in detail based on the provided context only. Please provide the most accurate response based on the question and if the answer is not in provided context just say, "answer to the question is not there in the context", don't provide the wrong answer.
    <context>
    {context}
    <context>
    Questions:{input}

    """
prompt = PromptTemplate(template=prompt_template, input_variables=["question"])

# @st.cache(allow_output_mutation=True)
def load_faiss_index(index_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(index_path,embeddings, allow_dangerous_deserialization=True)

# @st.cache(allow_output_mutation=True)
def load_documents(filepath):
    with open(filepath, 'rb') as f:
        documents = pickle.load(f)
    return documents


# def vector_embedding():
if "vectors" not in st.session_state:
    st.session_state.vectors = load_faiss_index('venv/faiss_index')
    st.session_state.documents = load_documents('venv/documents.pkl')
# prompt1=st.text_input("Enter Your Question From Documents")

# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

import time
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def chat_actions():
    st.session_state["chat_history"].append(
        {"role": "user", "content": st.session_state["chat_input"]},
    )

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": response,
            # st.write(response['answer'])
        }, 
    )

prompt1=st.chat_input("Enter your message", key="chat_input")
# use_fallback=False

if(prompt1):
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    print("Response time :",time.process_time()-start)
    response=retrieval_chain.invoke({'input':prompt1})['answer']

    # if ("The provided text does not contain any information" in response):
    #     print("Fallback to general model.")
    #     response = model.generate_content(prompt1).text
    chat_actions()


for i in st.session_state["chat_history"]:
    with st.chat_message(name=i["role"]):
        st.write(i["content"])




