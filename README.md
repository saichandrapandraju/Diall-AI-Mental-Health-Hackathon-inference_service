# Diall-AI-Mental-Health-Hackathon-inference_service

This repository contains the inference service app for transcription and summarization tasks. To run these models and start this service locally, follow these steps - 

1. Past your HuggingFace Access token in the line-28 of `app.py`
2. Install the requirements - `pip install -r requirements.txt`
3. Install `ffmpeg` by running - `apt-get install ffmpeg` or `brew install ffmpeg`
4. Start the app by running - `uvicorn app:app --host 0.0.0.0 --port 8000`

Your application is now running and you can access the swagger documentation of these APIs at `http://localhost:8000/docs/`
