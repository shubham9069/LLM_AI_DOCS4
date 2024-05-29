# LLM_AI_DOCS4 (Document Analysis and Question Answering)

# Live production deployment on Azure for use cases:-
```diff
**api for upload document to train you model :-**


URL:- http://azuremachine.koreacentral.cloudapp.azure.com:5000/upload-document<br>
payload : docs: fileName.pdf (it should be pdf and file formate should be formData)<br>
return :{<br>
file_name :"993863-SDE.pdf  // put this file name into a below API<br>
}<br>

**api for ask question to Ai :-**

URL:- http://azuremachine.koreacentral.cloudapp.azure.com:5000/ask-question<br>
payload : {<br>
  "question":"what is name of candidate ", // question to ask <br>
  "document":true, // id its true then Ai will use a user data . if its false its use a global information source  <br>
  "file_name":"993863-SDE.pdf" // if document:true the please mention the upload file name. in above api  file name are present in response <br>
}<br>
```



# Pre-requisite:
python >=3.9

## Description:
This project leverages GPT-4 along with various tools and models such as Langchain for dependency parsing, FaaIse vector DB for semantic similarity, Google LLM model for language understanding, and Hugging Face for embeddings. It provides a robust system for document analysis and question answering.

## Steps to Run:
1. **Set Up Virtual Environment**: Create a virtual environment for managing dependencies.
2. **Use Google API Key**: Obtain a Google API key for accessing the Google LLM model.
3. **Run Python Server**: Start the Python server to initialize the backend.
4. **Upload Document API**: Hit the upload-document API to send documents in PDF format. Note: Initial load time may be longer as models are pulled from Hugging Face.
5. **Ask Question API**: Utilize the ask-question API to ask questions related to the uploaded document or individual questions based on prompt templates.

## Usage:
1. Clone the repository.
2. Set up a virtual environment and activate it.
3. Install dependencies using `pip install -r requirements.txt`.
4. Obtain a Google API key and replace it in the appropriate configuration file.
5. Run the Python server.
6. Interact with the APIs as described above.


