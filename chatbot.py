# msu_ra_chatbot_app.py
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()  # load OPENAI_API_KEY from .env

# ---------------------------
# LangChain imports
# ---------------------------
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Attention is all you need", layout="wide")
st.title("Attention 👀......is all you need")

# Keep chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# Load local PDFs
# ---------------------------
pdf_folder = "./"  # same folder as script
all_docs = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, filename))
        docs = loader.load()
        all_docs.extend(docs)

# Concatenate all PDF content
string_list_concat = "".join([doc.page_content for doc in all_docs])

# ---------------------------
# Split by Markdown headers
# ---------------------------
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Section Title"), ("##", "Lecture Title")]
)
docs_list_md_split = md_splitter.split_text(string_list_concat)

# Ensure metadata exists
for doc in docs_list_md_split:
    if "Section Title" not in doc.metadata:
        doc.metadata["Section Title"] = "Unknown Section"
    if "Lecture Title" not in doc.metadata:
        doc.metadata["Lecture Title"] = "Unknown Lecture"

# ---------------------------
# Split into smaller chunks
# ---------------------------
token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=50
)
docs_list_tokens_split = token_splitter.split_documents(docs_list_md_split)

# ---------------------------
# Create embeddings and vectorstore
# ---------------------------
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=docs_list_tokens_split, embedding=embedding)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.7})

# ---------------------------
# LLM and Prompts
# ---------------------------
chat = ChatOpenAI(model_name="gpt-4o-mini", seed=365, temperature=0)
str_output_parser = StrOutputParser()

PROMPT_CREATING_QUESTION = '''Question:
{question_body}'''

prompt_creating_question = PromptTemplate.from_template(
    template=PROMPT_CREATING_QUESTION
)

PROMPT_RETRIEVING_S = '''You will receive a question from a student taking the course. 
Answer the question using only the provided context.
At the end of your response, include the section and lecture names where the context was drawn from, formatted as follows: 
'''

PROMPT_TEMPLATE_RETRIEVING_H = '''This is the question:
{question}

This is the context:
{context}'''

prompt_creating_question = PromptTemplate.from_template(template=PROMPT_CREATING_QUESTION)
prompt_retrieving_s = SystemMessage(content=PROMPT_RETRIEVING_S)
prompt_template_retrieving_h = HumanMessagePromptTemplate.from_template(template=PROMPT_TEMPLATE_RETRIEVING_H)
chat_prompt_template_retrieving = ChatPromptTemplate([prompt_retrieving_s, prompt_template_retrieving_h])

# ---------------------------
# Format context function (safe)
# ---------------------------
@chain
def format_context(dictionary):
    retrieved_list = dictionary["context"]
    formatted_string = ""
    for i, doc in enumerate(retrieved_list):
        formatted_string += f"{doc.page_content}\n\n-------------------\n"
    
    return {"context": formatted_string, "question": dictionary["question"]}


# ---------------------------
# RAG Chain
# ---------------------------
chain_retrieving_improved = (
    prompt_creating_question
    | RunnableLambda(lambda x: x.text)
    | {"context": retriever, "question": RunnablePassthrough()}
    | format_context
    | chat_prompt_template_retrieving
    | chat
    | str_output_parser
)

# ---------------------------
# Streamlit Input
# ---------------------------
user_question = st.text_area("Ask me something about 'Attention is all you need'", "")

if st.button("Get Answer") and user_question.strip() != "":
    result = chain_retrieving_improved.invoke({
        "question_body": user_question
    })

    st.markdown("### 💬 Answer:")
    st.write(result)

# Show chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📝 Chat History")
    for msg in st.session_state.chat_history:
        st.write(msg)

# ---------------------------
# Streamlit Sidebar: Sample Questions
# ---------------------------
# ---------------------------
# Streamlit Sidebar: Sample Questions
# ---------------------------

# Sidebar sample questions
st.sidebar.markdown("### 💡 Sample Questions")
sample_questions = [
    "What is the self-attention mechanism?",
    "How does positional encoding work?",
    "Explain multi-head attention in Transformers.",
    "What are the advantages of attention over RNNs?",
    "Why is 'Attention is all you need' significant?"
]

for idx, q in enumerate(sample_questions):
    if st.sidebar.button(q, key=f"sidebar_q{idx}"):
        user_question = q
        # Directly invoke RAG chain
        result = chain_retrieving_improved.invoke({"question_body": user_question})
        
        # Display answer immediately
        st.markdown("### 💬 Answer:")
        st.write(result)


