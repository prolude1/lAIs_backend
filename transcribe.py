import os
import sys
import whisper
from moviepy.editor import VideoFileClip
import yt_dlp

# currently not in use
def save_transcript(text,videoid):
    text_file = open("transcript/"+videoid+'.txt', "w")
    n = text_file.write(text)
    text_file.close()
# currently not in use
def read_transcript(videoid):
    path="transcript/"+videoid+'.txt'
    if os.path.isfile(path):
        f = open(path, "r")
        return f.read()
    else:
        return None


def extract_audio_from_video(filename, ext):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    clip = VideoFileClip(filename+ext)
    clip.audio.write_audiofile(f"{filename}.mp3")

def get_transcript_from_video(vf):
    filename, ext = os.path.splitext(vf)
    extract_audio_from_video(filename, ext)
    model = whisper.load_model("base")
    result = model.transcribe(f"{filename}.mp3")
    save_transcript(result["text"],filename)
    return result["text"]

def get_transcript_from_audio(sf):
    filename, _ = os.path.splitext(sf)
    model = whisper.load_model("base")
    result = model.transcribe(sf)
    save_transcript(result["text"],filename)
    return result["text"]

def get_transcript_from_url(url):
    ydl_opts = {
        'format': 'mp3/bestaudio/best',
        'outtmpl':'audio/%(id)s.%(ext)s',
        'keepvideo':True,
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([url])
    array=url.split("v=")
    array=array[1].split("&")
    sf=f"audio/{array[0]}.mp3"
    model = whisper.load_model("base")
    result = model.transcribe(sf)
    save_transcript(result["text"],array[0])
    return result["text"]