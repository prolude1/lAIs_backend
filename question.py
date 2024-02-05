from typing import List, Optional

import fire
import json
import os
from summarize import read_summary

from llama import Llama, Dialog





def savequestions(videoid,array):
    with open(f"question/{videoid}.json", "w") as fp:
        json.dump(array, fp)

def loadquestions(videoid):
    if not os.path.exists(f"question/{videoid}.json"):
        return None
    with open(f"question/{videoid}.json", "r") as fp:
        array=json.load(fp)
    return array

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    video_id: str,
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
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    summary=read_summary(video_id)
    if summary==None:
        return ["No summary generated, generate summary first."]
    dialog=[{"role": "system", "content":f"Format the next prompt with the following format: Add a '*' in front of each questions generated.Only generate questions."}
            ,{"role":"user","content":f"Create a set of questions that tests conceptes based on the following lecture summary text: {read_summary(video_id)}"}]
    dialogs = [dialog]
    result = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    addsystemdialog=[{"role": f"{result[0]['generation']['role']}", "content": f"{result[0]['generation']['content']}"}]
    dialog.append(addsystemdialog[0])

    savequestions(video_id,result[0]['generation']['content'].replace('\n','').split("* "))

    print("\n==================================\n")






if __name__ == "__main__":
    fire.Fire(main)
