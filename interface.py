# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, make_response, json, request
import transcribe
from summarize import read_summary
from chat import readchathistory
import subprocess
import os
from flask_cors import CORS
import question
from time import sleep
# Flask constructor takes the name of 
# current module (__name__) as argument.

CKPT_DIR="llama-2-7b-chat/"
TOKENIZER_PATH="tokenizer.model"

def get_transcript_url(videoid):
    transcript=transcribe.read_transcript(videoid)
    while transcript==None:
        sleep(5)
        transcript=transcribe.read_transcript(videoid)
    return transcript

def create_json_response(data: dict):
    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

def save_userinput(videoid,userinput):
    text_file = open("chat/"+videoid+'_user.txt', "w")
    n = text_file.write(userinput)
    text_file.close()

def check_chat_history(videoid):
    return os.path.exists("chat/"+videoid+'.txt')

app = Flask(__name__)
CORS(app)


@app.route('/api/youtube/<videoid>/summary')
def get_summary(videoid):

    summary=read_summary(videoid)
    if summary!=None:
        data = { "summary": summary, "id": videoid }
        print(data)
        return create_json_response(data)
    
    get_transcript_url(videoid)

    print("Start summarizing")
    subprocess.run(["torchrun", "--nproc_per_node","1","summarize.py",
                    "--ckpt_dir",f"{CKPT_DIR}",
                    "--tokenizer_path",f"{TOKENIZER_PATH}",
                    "--video_id",f"{videoid}"]) 
    print("Summarization done.")
    summary=read_summary(videoid)
    data = { "summary": summary, "id": videoid }
    return create_json_response(data)


@app.route("/")
def get_home():
    return "<h1>Hello</h1>"


@app.route('/api/youtube/<videoid>/transcription')
def get_transcript(videoid):

    # Read if file exist
    transcription=transcribe.read_transcript(videoid)

    # do transcript using OpenAI Whisper
    if transcription==None:
        transcription=transcribe.get_transcript_from_url(f"https://www.youtube.com/watch?v={videoid}")
    data = { "transcription": transcription, "id": videoid }

    print("Done transcribing")

    return create_json_response(data)

@app.route('/api/youtube/<videoid>/questions')
def get_question(videoid):

    questions=question.loadquestions(videoid)
    if questions!=[]:
        data = { "questions": questions, "id": videoid }
        print(questions)
        return create_json_response(data)
    else:
        subprocess.run(["torchrun", "--nproc_per_node","1","question.py",
                    "--ckpt_dir",f"{CKPT_DIR}",
                    "--tokenizer_path",f"{TOKENIZER_PATH}",
                    "--video_id",f"{videoid}",
                    ]) 
        questions=question.loadquestions(videoid)
        print(questions)
        data = { "questions": questions, "id": videoid }
        return create_json_response(data)

@app.route('/api/youtube/<videoid>/chat')
def chat_bot(videoid):
    question = request.args.get('question', '')
    print(question)
    save_userinput(videoid,question)
    if check_chat_history(videoid)==False:
        get_summary(videoid)
    
    subprocess.run(["torchrun", "--nproc_per_node","1","chat.py",
                    "--ckpt_dir",f"{CKPT_DIR}",
                    "--tokenizer_path",f"{TOKENIZER_PATH}",
                    "--video_id",f"{videoid}",
                    ]) 

    data=readchathistory(videoid)
    reply=data[-1]["content"]
    data = { "id": videoid,"reply": reply  }
    return create_json_response(data)

# main driver function:
if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()