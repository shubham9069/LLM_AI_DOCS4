from flask import Flask, request
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import json
from datetime import datetime
from flask_cors import CORS, cross_origin


class JsonConverter:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


app = Flask(__name__)
CORS(app)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def get_document(filePath):
    text = ""
    loader = PyPDFLoader(filePath)
    pages = loader.load_and_split()
    for page in pages:
        text += page.page_content
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([text])
    return docs


def get_vector_embedding(array_of_text, filename):
    db = FAISS.from_documents(array_of_text, embeddings)
    db.save_local(f"vectorDB/{filename}")
    return db


def get_conversation_chain(question, file_name):
    # it form finding a similar data realted to question
    vector_store = FAISS.load_local(
        f"vectorDB/{file_name}", embeddings, allow_dangerous_deserialization=True
    )

    # vector_context = vector_store.similarity_search(question)
    # context = ""
    # for text in vector_context:
    #     context += text.page_content
    # print(context)
    llm = GoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    template = """
    Answer the question using the given {context} and {chat_history}  
    If questions are asked where there is no relevant context available, please answer from ypur own information source .
Do not say "Based on the information you provided, ..." or "I think the answer is...".
Just answer should be justify and in details. 

Question: {question}
Answer: 
"""

    prompt = PromptTemplate(
    template=template,
    input_variables=[ "context","chat_history","question"]
        )
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain({"question": question})


@app.route("/ask-question", methods=["POST"])
def hello():
    question = request.json["question"]
    document = request.json["document"] 
    if document == True:
        filename = request.json["file_name"]
    else:
        filename = "default.pdf"
    try:
        response = get_conversation_chain(question,filename)

    except Exception as e:
        print(e)
        return str(e)
    return response["answer"]


@app.route("/health-check", methods=["GET"])
def health_check():
    
    return json.dumps({"google_api": os.getenv("GOOGLE_API_KEY")})

@app.route("/upload-document", methods=["POST"])
def uploadDocument():
    docs_pdf = request.files["docs"]
    fileName = str(datetime.now().microsecond) + "-"+ docs_pdf.filename.replace(" ","_")       
    docs_pdf.save(os.path.join("document", fileName))
    path = f"{app.root_path}\document\{fileName}"
    # reading our document
    text = get_document(path)
    array_chunks = get_text_chunks(
        text,
    )
    get_vector_embedding(array_chunks, fileName)
        
    return json.dumps({"fileName":fileName})


if __name__ == "__main__":
    app.run(port="5000", debug=True)
