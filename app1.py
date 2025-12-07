from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    ViltProcessor, ViltForQuestionAnswering
)

from langchain_community.tools import DuckDuckGoSearchRun
from PIL import Image
import io

from transformers import pipeline
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import numpy as np
import os 

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
app = FastAPI(title="Visual Question Answering and Captioning API", version="1.0")
##app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# Load BLIP (captioning) model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load ViLT (VQA) model
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

from peft import PeftModel, PeftConfig

# Load base model first
base_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load your LoRA adapter
vilt_model_ = PeftModel.from_pretrained(base_model, "./vilt-lora-vqa-lora")
vilt_model_.eval()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
)
#facebook llm 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
answer_recieved = ""
ans = ""
question_asked = ""
category = ""
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Stability trick
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # stability
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def split_heads(x, num_heads):
    # Input x: (seq_len, d_model)
    seq_len, d_model = x.shape
    d_k = d_model // num_heads
    return x.reshape(seq_len, num_heads, d_k)

def combine_heads(x):
    # Input x: (seq_len, num_heads, d_k)
    seq_len, num_heads, d_k = x.shape
    return x.reshape(seq_len, num_heads * d_k)

def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads):
    """
    X    : Input sequence (seq_len, d_model)
    W_q, W_k, W_v : Weight matrices (d_model, d_model)
    W_o  : Output projection weight (d_model, d_model)
    """
    d_model = X.shape[1]
    d_k = d_model // num_heads

    # Linear projections
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)

    # Split into heads
    Q = split_heads(Q, num_heads)  
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    # Compute attention for each head
    attention_outputs = []
    for i in range(num_heads):
        qi = Q[:, i, :]   
        ki = K[:, i, :]
        vi = V[:, i, :]

        scores = np.dot(qi, ki.T) / np.sqrt(d_k)
        weights = softmax(scores)
        output = np.dot(weights, vi)
        attention_outputs.append(output)

    # Stack and combine heads
    stacked = np.stack(attention_outputs, axis=1)  # (seq_len, num_heads, d_k)
    combined = combine_heads(stacked)              # (seq_len, d_model)

    # Final linear projection
    output = np.dot(combined, W_o)  # (seq_len, d_model)

    return output


def generate_caption(image_bytes):
    global ans
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    ans = caption
    return caption

def get_vqa_answer(image_bytes, question: str):
    global question_asked
    global answer_recieved  # add this line
    global category
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Step 1: Detect the object for /agent
    static_question = "what is in the image?"
    static_input = vilt_processor(image, static_question, return_tensors="pt")
    static_output = vilt_model(**static_input)
    static_idx = static_output.logits.argmax(-1).item()
    category = vilt_model.config.id2label[static_idx]  # update global variable

    # Step 2: Answer the user question
    inputs = vilt_processor(image, question, return_tensors="pt")
    outputs = vilt_model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    answer = vilt_model.config.id2label[idx]
    answer_recieved = answer
    question_asked = question

    return answer

@app.post("/caption")
async def caption_endpoint(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        caption = generate_caption(image_bytes)
        return JSONResponse({"Caption": caption})
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.post("/vqa")
async def vqa_endpoint(image: UploadFile = File(...), question: str = None):
    try:
        if not question:
            return JSONResponse({"error": "Question must be provided."})
        image_bytes = await image.read()
        answer = get_vqa_answer(image_bytes, question)
        return JSONResponse({"Answer": answer, "DetectedObject": category})

    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.post("/agent")
async def agentcall(agentic_Search_on_the_above_image: str = None):
    try:
        template = """
You are a knowledgeable and imaginative assistant. Your task is to provide a rich and meaningful description of an image based solely on a short caption and the answer recieved when the user asked a question related to image.

Caption: "{ans}"
question asked: "{question_asked}"
answer recieved: "{answer_recieved}"

Instructions:
- Do NOT include phrases like "Based on the caption" or "I'm going to guess".
- Assume the role of someone who has visually seen the image and is describing it in a natural and confident tone.
- Feel free to infer context, emotion, setting, or background as long as it's plausible from the caption.
- Keep the language fluid and human-like.
- wrap up in 90-80 words

Begin your response now:
"""
        prompt = ChatPromptTemplate.from_template(template)
        formatted_prompt = prompt.format(ans=ans, question_asked=question_asked, answer_recieved=answer_recieved)
        summary = llm.invoke(formatted_prompt)
        return JSONResponse({"Answer from agent": summary.content})
    except Exception as e:
        return JSONResponse({"Sorry, please reach out to the Admin!": str(e)})
