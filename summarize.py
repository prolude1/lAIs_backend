from typing import List, Optional

import fire
import transcribe as transcribe

from llama import Llama, Dialog
import copy

import torch
from time import sleep
torch.cuda.set_device(2)
import os

# currently not in use
def save_summary(text,videoid):
    text_file = open("summary/"+videoid+'.txt', "w")
    n = text_file.write(text)
    text_file.close()
# currently not in use
def read_summary(videoid):
    path="summary/"+videoid+'.txt'
    if os.path.isfile(path):
        f = open(path, "r")
        return f.read()
    else:
        return None

def initialize_llama(ckpt_dir,tokenizer_path,max_seq_len=10000,max_batch_size=1):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator

def split_transcript(transcript,max_seq_len):
    resultarray=[]
    returnarrayofarray=[]
    returnarray=[]
    index=0
    transcriptarray=transcript.split('. ')
    no_sentences=int((max_seq_len)/100)
    for i in transcriptarray:
        if index>no_sentences:
            returnarrayofarray.append(copy.deepcopy(resultarray))
            resultarray=[]
            index=0
        resultarray.append(i)
        index+=1
    if len(resultarray)>0:
        returnarrayofarray.append(copy.deepcopy(resultarray))
    for j in returnarrayofarray:
        returnarray.append('. '.join(j))
    return returnarray

def summarize_transcript(ckpt_dir,tokenizer_path,transcript,max_seq_len=10000,max_batch_size=1,temperature=0.6,top_p=0.9,max_gen_len=500):
    model=initialize_llama(ckpt_dir,tokenizer_path,max_seq_len=max_seq_len,max_batch_size=max_batch_size)
    transcriptarray=split_transcript(transcript, max_seq_len)
    dialog=[]
    results=[]
    for i in transcriptarray:
        dialog=[[
            {
                "role":"user",
                "content": f"""
[summarisation]
As an expert in summarising, please summarise the following piece of text in less than 1500 words. Use formal language in the past tense, and format your summary in markdown. Bold words that are of importance. The text is as follows:

{i}

Summary:
"""
            },
            ]]
        
        result=model.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )

        results.append(result[0]['generation']['content'].strip())
                       
        # print("Part of transcript:\n", i)
        # print("")
        # print("Summarised result:\n", result[0]['generation']['content'].strip())
        # print("======================")
    print(results)
    summary="".join(results)
    dialog=[[
            {
                "role":"user",
                "content": f"""
[summarisation]
As an expert in summarising, please paraphrase the following piece of text in less than 1500 words in a coherent manner. Use formal language in the past tense, and format your summary in markdown. Bold words that are of importance. The text is as follows:

{summary}

Summary:
"""
            },
            ]]
        
    result=model.chat_completion(
        dialog,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p
    )
    f = open("final summary.txt", "w")
    f.write(f"Sumary:\n{summary}\n\nFinal Summary:\n{result[0]['generation']['content'].strip()}")
    f.close()
    print("\nFinal summary: \n")
    print(result[0]['generation']['content'].strip())
    return result[0]['generation']['content'].strip()

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    video_id: str = None,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 10000,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
    
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    if max_gen_len==None:
        max_gen_len=1000
    # transcript=transcribe.get_transcript_from_url("https://www.youtube.com/watch?v=ZK3O402wf1c")
        
    
    # f = open("transcript.txt", "w")
    # f.write(transcript)
    # f.close()
    
    transcript=transcribe.read_transcript(video_id)
    while transcript==None:
        sleep(5)
        transcript=transcribe.read_transcript(video_id)
    # transcribe.save_transcript(transcript,f"/transcript/{id}.txt")
    # id="52vW8V9srw8"
    # transcript=transcribe.read_transcript(f"/transcript/{id}.txt")
    summary=summarize_transcript(ckpt_dir,tokenizer_path,transcript,max_seq_len=max_seq_len,max_batch_size=max_batch_size,temperature=temperature,top_p=top_p,max_gen_len=max_gen_len)
    save_summary(summary,video_id)




if __name__ == "__main__":
    fire.Fire(main)



    