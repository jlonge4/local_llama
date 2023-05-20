# local_llama

Interested in chatting with your PDFs entirely offline and free from OpenAI dependencies? Then you're in the right place. I made my other project, gpt_chatwithPDF with the ultimate goal of local_llama in mind, this assumes the same functinoality as that project but is, local and can be run in airplane mode.. Drop a star if you like it!

DISCLAIMER: This is an experimental repo, not an end all be all for your solution. It is meant as a way forward towards many use cases for local offline use of LLMs.

## Installation

On windows you have to have Visual Studio with a C compiler installed. 
Secondly you need a model, I used GPT4All-13B-snoozy.ggml.q4_0.bin, however any ggml should work. 
I downloaded the model here https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML/tree/main
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
