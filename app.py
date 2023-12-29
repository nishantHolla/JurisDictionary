import sys
import os

import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.memory import StreamlitChatMessageHistory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

import streamlit as st
from streamlit_chat import message

if len(sys.argv) < 2:
    print("Usage: [PROGRAM_NAME] [openai_api_key]")
    sys.exit(1)

KEY = sys.argv[1]
os.environ["OPENAI_AI_KEY"] = KEY
openai.api_key = KEY

history = StreamlitChatMessageHistory(key="chat_messages")
loader = PyPDFLoader("./legalguide.pdf")
pages = loader.load_and_split()
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=50)
documents = text_splitter.split_documents(pages)
embeddings = OpenAIEmbeddings(openai_api_key=KEY)
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=KEY,
)


db = FAISS.from_documents(documents, embeddings)

memory = ConversationBufferMemory(
    llm=llm, output_key="answer", memory_key="chat_history", return_messages=True
)
retriever = db.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "include_metadata": True}
)

#
prompt_template = """
You are a world-class highly trained professional attorney who is an expert in all matters of the \
law in every country in the world.
You are highly qualified and can provide guidance and also advocate on the user's behalf.
Answer the questions like a lawyer would in a systematic way. Be elaborate, but give responses that\
are understandable by users who are not educated with the law.
You know everything there is to know about family law, criminal defense, corporate law, personal injury\
law, labour law, immigration law, tax law, bankruptcy law, \
environmental law, real estate law, constitutional law, patents, intellectual property and copyright law,\
divource law, civil rights law, public defence, \
maritime law and business law.
You are able to act as a barrister, solicitor, prosecutor, advocate and contract attorney when asked to do so.
When a user has admitted to any form of guilt, convey it to them and offer advice on how to go ahead.
You are friendly and try your hardest to calm a stressed user down while offering good advice.
You are able to calculate legal damages to be paid if asked to do so.
You are able to find legal loopholes in a given contract or document and point it out to the user.
Use the legal terms in your responses while still making it easy for the user to understand your answer.
Say I don't know if you don't know the answer to a question or if the answer lies outside your domain.
Context: \n {context}?\n
Question: \n {question}?\n

Answer:

"""

message = [
    SystemMessagePromptTemplate.from_template(prompt_template),
]
qa_prompt = ChatPromptTemplate.from_messages(message)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    get_chat_history=lambda h: h,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    verbose=False,
)

chat_history = []

st.set_page_config(
    page_title="JurisDictionary",
    page_icon="https://raw.githubusercontent.com/nishantHolla/JurisDictionary/main/botAvatar.jpg",
)


BotAvatar = (
    "https://raw.githubusercontent.com/nishantHolla/JurisDictionary/main/botAvatar.jpg"
)
UserAvatar = (
    "https://raw.githubusercontent.com/nishantHolla/JurisDictionary/main/userAvatar.png"
)


st.title("JurisDictionary")


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = [
        "Hello there! I'm JurisDictionary, your personal advocate. Have any doubts on the law? Got yourself in a sticky situation? Let me help!"
    ]
    message(
        "Hello there! I'm JurisDictionary, your personal advocate. Have any doubts on the law? Got yourself in a sticky situation? Let me help!",
        is_user=False,
        avatar_style="no-avatar",
        logo=BotAvatar,
    )
if "past" not in st.session_state:
    st.session_state["past"] = ["Hey!"]


def generate_response(query):
    result = chain(
        {"question": query, "chat_history": st.session_state["chat_history"]}
    )
    st.session_state["chat_history"] = [(query, result["answer"])]
    return result["answer"]


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    message(m["prompt"], is_user=m["is_user"], logo=m["avatar"])

if query := st.chat_input("Ask away."):
    message(query, is_user=True, avatar_style="no-avatar", logo=UserAvatar)
    result = chain({"question": query, "chat_history": chat_history})
    message(result["answer"], is_user=False, avatar_style="no-avatar", logo=BotAvatar)
    history.add_user_message(query)
    history.add_ai_message(result["answer"])

    st.session_state.messages.append(
        {"prompt": query, "is_user": True, "avatar": UserAvatar}
    )
    st.session_state.messages.append(
        {"prompt": result["answer"], "is_user": False, "avatar": BotAvatar}
    )
