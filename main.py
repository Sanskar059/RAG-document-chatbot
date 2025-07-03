import streamlit as st
import time
import fitz # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
# vector embeddings of the pdf text
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# creating a qudrant store to store the vector embeddings
from langchain_qdrant import QdrantVectorStore
import json

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_qdrant import QdrantVectorStore
st.title("📚 Document Chatbot")
pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file is not None:
    st.success(f"Uploaded file: {pdf_file.name}")

    # Step 2: Read in-memory bytes
    pdf_bytes = pdf_file.read()

    # Step 3: Open with PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Step 4: Convert pages to LangChain Document objects
    document = [
        Document(
            page_content=page.get_text(),
            metadata={"page": i + 1}
        )
        for i, page in enumerate(doc)
    ]


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(document)



    google_api_key = os.getenv("GOOGLE_API_KEY")


    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
       )



    vector_store = QdrantVectorStore.from_documents(
    documents=texts,
    collection_name="Implementing_vector",
    embedding=embeddings,
    url="http://localhost:6333",
    prefer_grpc=False
    # Important for local instances
)


    st.title("Wait AI is reading Your Document")

    with st.spinner("Loading... please wait!"):
     time.sleep(3)  # Simulate slow processing

    st.success("Done!")
    


    vector_db = QdrantVectorStore.from_existing_collection(
   
    url="http://localhost:6333",
     collection_name="Implementing_vector",
      embedding=embeddings,
    
)
    client = OpenAI(
    api_key="google_api_key",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

    if "messages" not in st.session_state:
      st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
     ]

     # Render chat history in UI
    for msg in st.session_state.messages:
      if msg["role"] != "system":  # Optional: skip showing system prompt
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about your document!", key="chat_input_main")

    if query:
    # Add user message to history
      st.session_state.messages.append({"role": "user", "content": query})
      with st.chat_message("user"):
        st.markdown(query)
      with st.spinner("Searching documents..."):
       search_result = vector_db.similarity_search(
       query = query
     )
 
       context = "\n\n\n".join([f"page content: {result.page_content}\nPage Number: {result.metadata['page']}" for result in search_result])
       SYSTEM_PROMPT= f"""
You are healpful Assistant. Answers the users query on based on the context provided below.
use simple text format. Do not use JSON or objects. 
Give a nicely formatted answer with headings and bullet points.
Reffer to the page number also to get more information about the topic.

recive the context from the page content and also provide page number to user for further refrence.

Output format : 
{{ "content" : "string"}}
INSTRUCTION:
- Remove 'chapter x' from your response.
- If the user asks for 'more detail' or similar, expand and elaborate more.
- Remove any page or chapter number written *inside* the page content itself.
- Provide the page number on the last line of your response.
-if content is on multiple page mention all page numbers
Example - content \n\n for mre info you can refer to page number 7 , 17
Context:
{context}
"""


       messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
       ]

      with st.spinner("Generating answer..."):
        response = client.chat.completions.create(
        model="gemini-2.0-flash",
        response_format={"type": "json_object"},
        messages=messages
    )

      content = response.choices[0].message.content
      messages.append({"role": "assistant", "content": content})

      data = json.loads(content)
      answer = data.get('content')
      st.session_state.messages.append({"role": "assistant", "content": answer})
      with st.chat_message("assistant"):
        st.markdown(answer)

    