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

## History


## Credits

The-Bloke and his model GPT4All-13B-snoozy.ggml.q4_0.bin that I used for this project.

## License

Apache 2.0
