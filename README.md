# local_llama

Interested in chatting with your PDFs, TXT files, or Docx files entirely offline and free from OpenAI dependencies? Then you're in the right place. I made my other project, gpt_chatwithPDF with the ultimate goal of local_llama in mind. This repo assumes the same functionality as that project but is local and can be run in airplane mode.. Drop a star if you like it!

Video demo here: https://www.reddit.com/user/Jl_btdipsbro/comments/13n6hbz/local_llama/?utm_source=share&utm_medium=ios_app&utm_name=ioscss&utm_content=2&utm_term=1

DISCLAIMER: This is an experimental repo, not an end all be all for your solution. It is meant as a way forward towards many use cases for local offline use of LLMs.

## Installation

On windows you have to have Visual Studio with a C compiler installed. 
Secondly you need a model, I used llama-2-7b-chat.Q4_K_M.gguf, however any gguf should work. 

* Note GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp.

I downloaded the model here https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
Run pip install -r requirements.txt

## Usage

Use the command python -m streamlit run "path/to/project/local_llama.py". This will start the app in your browser. Once you have uploaded your PDFs, refresh the browser, select a manual, and ask away!

CLI output as an example for inference time running on my alienware x14 with 3060:

|                                     TIMES                                                   |
| :-----------------------------------------------------------------------------------------: |
|/GPT4All-13B-snoozy.ggml.q4_0.bin                                                            |
|llama_model_load_internal: format     = ggjt v2 (latest)                                     |
|llama_model_load_internal: n_vocab    = 32000                                                |
|llama_model_load_internal: n_ctx      = 512                                                  |
|llama_print_timings:        load time = 21283.78 ms                                          |
|llama_print_timings:      sample time =     3.08 ms /    13 runs   (    0.24 ms per token)   |
|llama_print_timings: prompt eval time = 21283.70 ms /   177 tokens (  120.25 ms per token)   |
|llama_print_timings:        eval time =  2047.03 ms /    12 runs   (  170.59 ms per token)   |
|llama_print_timings:       total time = 24057.21 ms                                          |


## History


## Credits

The-Bloke and his model GPT4All-13B-snoozy.ggml.q4_0.bin that I used for this project.

## License

Apache 2.0
