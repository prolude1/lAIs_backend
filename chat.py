from typing import List, Optional

import fire
import json
import os
from summarize import read_summary

from llama import Llama, Dialog


def readinput(videoid):
    path="chat/"+videoid+'_user.txt'
    if os.path.isfile(path):
        f = open(path, "r")
        return f.read()
    else:
        return None

def readchathistory(videoid):
    path="chat/"+videoid+'.json'
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        return []


def savechathistory(videoid,list_of_dict):
    path="chat/"+videoid+'.json'
    with open(path, 'w') as fout:
        json.dump(list_of_dict, fout)

def clear_dialog(dialog):
    print(len(dialog))
    if len(dialog)>10:
        final_dialog=[dialog[0]]
        final_dialog.extend(dialog[-9:])
        return final_dialog
    else:
        return dialog

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
    dialog=readchathistory(video_id)
    userinput=readinput(video_id)
    check=False
    if userinput==None:
        userinput=input("Chat With Me:")
        check=True
    while True:
        if dialog==[]:
            summary=read_summary(video_id)
            dialog=[{"role": "system", "content":f"Answer all prompts based on the context of the following lecture summary: {read_summary(video_id)}"}]
        adduserdialog={"role": "user", "content": f"{userinput}"}
        dialog=clear_dialog(dialog)
        dialog.append(adduserdialog)
        dialogs = [dialog]
        result = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        addsystemdialog=[{"role": f"{result[0]['generation']['role']}", "content": f"{result[0]['generation']['content']}"}]
        dialog.append(addsystemdialog[0])
        print(
            f"> {result[0]['generation']['role'].capitalize()}: {result[0]['generation']['content']}"
        )
        print("\n==================================\n")
        savechathistory(video_id,dialog)
        os.remove("chat/"+video_id+'_user.txt')
        if check==False:
            break



if __name__ == "__main__":
    fire.Fire(main)
