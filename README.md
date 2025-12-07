# Vision‑Question Answering & Captioning API

A FastAPI‑based web API that lets you upload an image and either:  
- get a natural‑language caption describing the image, or  
- ask a question about the image (via VQA — Visual Question Answering) and get an answer.  

The API uses state‑of‑the‑art vision‑language transformer models from the Hugging Face / Transformers ecosystem.

---

## Features

- **Image Captioning** — Automatically generate a caption that describes the contents of an image.  
- **Visual Question Answering (VQA)** — Ask natural-language questions about an image and get relevant answers.  
- **Simple HTTP API** — Upload image via REST endpoints. Easy to integrate with web or mobile front‑ends.  
- **Pre-trained Models** — Uses powerful pre-trained models (captioning + VQA), so you don’t need to train from scratch.  
- **Extensible Architecture** — Easily extendable to support more models / tasks (e.g., image‑to‑text, advanced VQA, multi‑modal inputs).

---

##  Tech Stack & Dependencies

- Backend framework: FastAPI — for API endpoints and routing. :contentReference[oaicite:1]{index=1}  
- Vision‑language models: BLIP (for image captioning) and ViLT (for VQA), via Hugging Face Transformers.  
- Image handling: Pillow (PIL) for image loading and conversion.  
- Environment & configuration management: `.env` for storing keys / configs (if needed).  
- Optional: Additional LLM‑based agent for further image-based reasoning or context enrichment (configurable).  

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName
