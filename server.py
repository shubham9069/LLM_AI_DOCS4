from flask import Flask, request
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
import os
from werkzeug.utils import secure_filename
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from diffusers import StableDiffusionPipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import torch
import json


class JsonConverter:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


app = Flask(__name__)

genai.configure(api_key="AIzaSyBc22wVVpuwKM3FK0zqqcOvuNbCV1eGi2Q")
model = genai.GenerativeModel("gemini-pro")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vector_store = ""


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


def get_conversation_chain(question):
    # it form finding a similar data realted to question

    # vector_context = vector_store.similarity_search(question)
    # context = ""
    # for text in vector_context:
    #     context += text.page_content

    llm = GoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key="AIzaSyBc22wVVpuwKM3FK0zqqcOvuNbCV1eGi2Q",
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


def get_image_generatiion(prompt):

    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to("cuda")
    image = pipe("a photo of an astronaut riding a horse on mars").images[0]

    image.save("astronaut_rides_horse.png")
    return ""


@app.route("/ask-question", methods=["POST"])
def hello():
    question = request.json["question"]
    response = get_conversation_chain(question)
    print(response)
    return response["answer"]


@app.route("/image-prompt", methods=["POST"])
def tentToImage():
    prompt = request.json["prompt"]
    response = get_image_generatiion(prompt)
    return response


@app.route("/upload-document", methods=["POST"])
def uploadDocument():
    global vector_store
    docs_pdf = request.files["docs"]
    docs_pdf.save(os.path.join("document", secure_filename(docs_pdf.filename)))
    path = f"{app.root_path}\document\{secure_filename(docs_pdf.filename)}"
    
    # reading our document
    text = get_document(path)
    array_chunks = get_text_chunks(
        text,
    )
    vector_store = get_vector_embedding(
        array_chunks, secure_filename(docs_pdf.filename)
    )
    return "document upload successfully"


if __name__ == "__main__":
    app.run(port="5000", debug=True)
