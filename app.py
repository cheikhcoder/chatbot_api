from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import os
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load Wolof ↔ French translation model
model_name = "cifope/nllb-200-wo-fr-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Initialize FastAPI app
app = FastAPI()

# Global variables
vectorstore = None
chatbot_instance = None  # Store chatbot globally

# Translation functions
def translate_to_wolof(text):
    tokenizer.src_lang = "fra_Latn"
    tokenizer.tgt_lang = "wol_Latn"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    result = model.generate(**inputs.to(model.device), max_new_tokens=512)
    return tokenizer.batch_decode(result, skip_special_tokens=True)[0]

def translate_from_wolof(text):
    tokenizer.src_lang = "wol_Latn"
    tokenizer.tgt_lang = "fra_Latn"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    result = model.generate(**inputs.to(model.device), max_new_tokens=512)
    return tokenizer.batch_decode(result, skip_special_tokens=True)[0]

# PDF text extraction and vectorstore creation
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20, separator="\n", length_function=len)
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_chatbot(vectorstore):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# Initialize chatbot with files in "dossiers"
def initialize_chatbot():
    global vectorstore, chatbot_instance
    pdf_directory = "dossiers"
    pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
    if pdf_files:
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        chatbot_instance = get_chatbot(vectorstore)
        return True
    return False

# FastAPI startup event to initialize chatbot
@app.on_event("startup")
async def startup_event():
    success = initialize_chatbot()
    if not success:
        print("No PDF files found in 'dossiers' directory.")

# Chat endpoint
@app.post("/chat/")
async def chat(question: str = Form(...), language: str = Form("Français")):
    global chatbot_instance
    try:
        if not chatbot_instance:
            return JSONResponse(content={"error": "Chatbot is not initialized. Please ensure PDF files are available in the 'dossiers' directory."}, status_code=400)

        if not question:
            return JSONResponse(content={"error": "Question cannot be empty"}, status_code=400)

        # Translate Wolof input to French if needed
        if language == "Wolof":
            question = translate_from_wolof(question)

        # Use chatbot to generate response
        response = chatbot_instance({'question': question})
        response_text = response['answer']

        # Translate response to the selected language
        if language == "Wolof":
            response_text = translate_to_wolof(response_text)
        elif language != "Français":
            response_text = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": f"Translate to {language}: {response_text}"}]
            )['choices'][0]['message']['content']

        return JSONResponse(content={"response": response_text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
