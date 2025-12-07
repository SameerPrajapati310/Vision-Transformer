from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    pipeline
)
from langchain_community.tools import DuckDuckGoSearchRun
from PIL import Image
import io

app = FastAPI(title="Visual Question Answering and Captioning API", version="2.0")

# Load BLIP (Captioning model)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load BLIP-2 (Visual Question Answering model)
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Load Summarization Pipeline (Facebook BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")



@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

def generate_caption(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def get_vqa_answer_blip2(image_bytes, question: str):
    global category
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Detect main object (auto question)
    static_prompt = "What is in the image?"
    static_inputs = blip2_processor(images=image, text=static_prompt, return_tensors="pt")
    static_output = blip2_model.generate(**static_inputs, max_new_tokens=50)
    category = blip2_processor.batch_decode(static_output, skip_special_tokens=True)[0].strip()

    # User question
    user_inputs = blip2_processor(images=image, text=question, return_tensors="pt")
    user_output = blip2_model.generate(**user_inputs, max_new_tokens=50)
    answer = blip2_processor.batch_decode(user_output, skip_special_tokens=True)[0].strip()

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
        answer = get_vqa_answer_blip2(image_bytes, question)
        print(answer)
        return JSONResponse({"Answer": answer, "DetectedObject": category})
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.post("/agent")
async def agentcall(agentic_Search_on_the_above_image: str = None):
    try:
        search_tool = DuckDuckGoSearchRun()
        result = search_tool.invoke(f'tell about {category}')
        summary = summarizer(result, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        return JSONResponse({"Answer from agent": summary})
    except Exception as e:
        return JSONResponse({"Sorry, please reach out to the Admin!": str(e)})
