
#### Step 1: request permission for the Llama-2–7B model
* Hugging face approval: https://huggingface.co/meta-llama/Llama-2-7b-hf
* Meta approval: https://ai.meta.com/resources/models-and-libraries/llama-downloads/
* WRITE access token: https://huggingface.co/settings/tokens)
* Execute huggingface-cli login


#### Step 2: Prerequisites
```
pip install transformers --upgrade (to 4.38.0)
pip install trl==0.4.7
pip install peft==0.4.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.3
```
