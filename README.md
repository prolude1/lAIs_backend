To set up, install python 3.9 and set up an environment. Ensure that WSL or Linux is available.

Install llama and download the 7b-chat model. The model can be downloaded from Facebook. Request download link from Facebook. Ensure ffmpeg is installed.The code here specifically uses llama-2-7b-chat.

conda create -n <env_name>
pip3 install openai-whisper moviepy python-ffmpeg flask yt-dlp flask-cors
conda install --file requirements.txt


The backend handles two models, mainly OpenAI Whisper and Llama 2. 

OpenAI Whisper is mainly used to transcribe using a audio clip. 

The weights used in Llama 2 is provided by Meta and needs to be downloaded from Meta. 7b-chat is used in this case.

A summary can then be generated using Llama 2. In addition, Llama 2 is used as a chatbot in this case to help students answer questions relating to the topic in the lecture. Llama 2 also serves as a question provider for students who would like to refresh or strengthen their knowledge of the topic after understandingg the lecture.