from fastapi import FastAPI
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch
import re
import whisper
import json
from tqdm import tqdm
import os
from transformers import pipeline as tr_pipeline
import logging
from fastapi import HTTPException
import shutil
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    force=True,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

app = FastAPI()
# paste your access token
access_token = ""
whisper_size = "base"
pyannotate_model = "pyannote/speaker-diarization-3.0"
conv_summarize_model = "philschmid/bart-large-cnn-samsum"

# Load the trained models
pipeline = Pipeline.from_pretrained(pyannotate_model, use_auth_token=access_token)
# pipeline.to(device)
whisper_model = whisper.load_model(whisper_size, device="cpu")
conversation_summarizer = tr_pipeline(
    "summarization", model=conv_summarize_model, device=device
)


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s


@app.get("/")
def read_root():
    return {"message": "Welcome to the Transcription API"}


@app.post("/transcribe/")
def transcribe(data: dict) -> dict:
    """
    Given the mp3/wav file of the conversation, returns the transcribed text. input json is expected to be in the form - {"recording_path":`recording file path`}
    Result will be in the form - {"transcription":`transcribed text`}
    """
    logging.info(f"received '/transcribe/' request with data - {data}")
    recording_path = data["recording_path"]
    format = recording_path.split(".")[-1]
    if format not in ["mp3", "wav"]:
        return HTTPException(
            "404", "Only '.mp3' or '.wav' formats are supported for the recording."
        )

    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    os.chdir(tmp_dir)
    # "input_prep.wav" = os.path.join(tmp_dir, "input_prep.wav")
    # "diarization.txt" = os.path.join(tmp_dir, "diarization.txt")
    os.system(
        f"ffmpeg -i {repr(recording_path)} -vn -acodec pcm_s16le -ar 16000 -ac 1 -y input.wav"
    )
    time.sleep(2)
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = AudioSegment.from_wav("input.wav")
    audio = spacer.append(audio, crossfade=0)
    audio.export("input_prep.wav", format="wav")

    logging.info(
        f"Running diarization. On average, this'll take ~2.5 minutes for a 5 minute recording."
    )

    dz = pipeline({"uri": "None", "audio": "input_prep.wav"}, num_speakers=2)
    with open("diarization.txt", "w") as text_file:
        text_file.write(str(dz))

    doc_ = list(dz.itertracks(yield_label=True))[0][-1]
    if doc_ == "SPEAKER_00":
        speakers_ = {"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}
    else:
        speakers_ = {"SPEAKER_00": "Patient", "SPEAKER_01": "Doctor"}

    dzs = open("diarization.txt").read().splitlines()

    groups = []
    g = []
    lastend = 0

    for d in dzs:
        if g and (g[0].split()[-1] != d.split()[-1]):  # same speaker
            groups.append(g)
            g = []

        g.append(d)

        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=d)[1]
        end = millisec(end)
        if lastend > end:  # segment engulfed by a previous segment
            groups.append(g)
            g = []
        else:
            lastend = end
    if g:
        groups.append(g)

    audio = AudioSegment.from_wav("input_prep.wav")
    gidx = -1
    for g in groups:
        start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
        start = millisec(start)  # - spacermilli
        end = millisec(end)  # - spacermilli
        gidx += 1
        audio[start:end].export(str(gidx) + ".wav", format="wav")

    # Whisper
    logging.info(
        "Running whisper to transcribe. On average, this will take ~50 seconds for a 5 minute recording."
    )
    for i in tqdm(range(len(groups)), desc="Transcribing..."):
        audiof = str(i) + ".wav"
        result = whisper_model.transcribe(
            audio=audiof, language="en", word_timestamps=True
        )
        with open(str(i) + ".json", "w") as outfile:
            json.dump(result, outfile, indent=4)

    speakers = {
        "SPEAKER_00": (speakers_["SPEAKER_00"], "#e1ffc7", "darkgreen"),
        "SPEAKER_01": (speakers_["SPEAKER_01"], "white", "darkorange"),
    }

    txt = list("")
    gidx = -1
    for g in groups:
        shift = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        shift = millisec(shift) - spacermilli  # the start time in the original video
        shift = max(shift, 0)
        gidx += 1
        captions = json.load(open(str(gidx) + ".json"))["segments"]
        if captions:
            speaker = g[0].split()[-1]
            if speaker in speakers:
                speaker, _, _ = speakers[speaker]
            for c in captions:
                start = shift + c["start"] * 1000.0
                start = start / 1000.0  # time resolution ot youtube is Second.
                end = (shift + c["end"] * 1000.0) / 1000.0
                if txt and txt[-1].split(":")[0].strip() == f"[{speaker}]":
                    txt[-1] += f'{c["text"]}'
                else:
                    txt.append(f'\n[{speaker}]: {c["text"]}')

    # out_file = os.path.join(os.getcwd(), "transcription.txt")
    # with open(out_file, "w", encoding='utf-8') as file:
    #     s = "".join(txt)
    #     file.write(s)
    #     print(f'captions saved to {out_file}')
    #     print(s+'\n')
    # return {"out_file": out_file}
    os.chdir("../")
    shutil.rmtree(tmp_dir)

    return {"transcription": "".join(txt).strip()}


@app.post("/summarize/")
def summarize(data: dict) -> dict:
    """
    Given the transcribed conversation between patient and doctor or patient's progress tracking, returns a short summary in the form - {"summary_text":`summary`}.
    Input json is expected to be in the form - {"text":`text to summarize`}
    """
    logging.info("Summarizing the conversation. On average, this will take ~5 seconds.")
    return conversation_summarizer(data["text"])[0]
