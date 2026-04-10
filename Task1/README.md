cynaptics induction task 1 - GPT-2 trained on shakespeare text
this is my submission for the cynaptics club induction task 1 . I built a decoder-only transformer model from scratch in pytorch to generate shakespeare text

architecture details
parameters: around 14.5M parameters . is 10M parameters a lot ? for a 1MB dataset it is a bit oversized but I kept it so the model had the capacity to learn complex stylistic nuances instead of just shrinking the architecture

layers & heads: 6 transformer blocks with 6 attention heads and 384 embedding dimension

activation: used GELU intead of the standard ReLU for smoother gradient flow

tokenizer and the 12k vocab trade off
I trained a custom Byte-Pair Encoding (BPE) tokenizer . initially I tried a vocab_size of 301 but it kept splitting words into sub-word fragments like 'd es' .
I increased the vocab_size to 12000 . at vocab_size of 12000 I found that it had picked complex words perfectly without fragmentation so I am keeping this vocab_size of 12000 .

the architectural trade off is that 12000 tokens on a tiny 1MB dataset creates a sparse transition matrix . the model formats things perfectly and uses great vocabulary like 'chamber' and 'disease' but sometimes struggles with the grammatical logic because it hasn't seen the rare words enough times . I used a dropout of 0.3 and weight decay of 0.0 to keep the logits sharp while preventing it from just overfitting and memorizing the dataset

formatting and newlines
why is the model not currently learning the newline character after each character name ? because the cross-entropy loss prioritizes semantic meaning over syntax . instead of retraining the entire tokenizer and model from scratch with a special newline token today only and risking breaking the tensor shapes , I added a regex post-processing function in the inference loop . it intercepts the raw decoded string and automatically formats the script with newlines after character names . this separates the neural network logic from the UI presentation

training and early stopping
I trained this on my local gpu from the terminal . I used the AdamW optimizer with a learning_rate of 1e-4 so it learns slowly and carefully .
I implemented early stopping with a patience limit . what is the val loss why is it important to monitor it more than the training loss ? because training loss just shows memorization . my code tracks the val loss and saves the best weights before the gap between train and val starts diverging , ensuring the model actually generalizes

references and sites used
Andrej Karpathy's "Let's build GPT" (YouTube): used this as the core foundation to understand the math behind multi-head self-attention , causal masking , and the overall gpt block structure

PyTorch Documentation: used for implementing nn.Module , nn.Embedding , and the AdamW optimizer

HuggingFace Tokenizers Library: used the documentation to figure out how to train and implement the custom BPE tokenizer from scratch

## References and resources used :  
1. The main structure of the code follows the code provided by **Andrej Karpathy** in his video on gpt-2 and the corresponding repo for the same video <br>
   https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4698s <br>
   https://github.com/karpathy/ng-video-lecture

2. I watched the pytorch basics tutorials from the following playlist from **Patrick Loeber** <br>
   https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

3. I used the tokenizer BPE that is provided by hugging face in their tokenizers library <br>
   https://huggingface.co/docs/tokenizers/quicktour

4. The 3B1B videos helped me get the intuition of a Transformer <br>
   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

5. I also used wikipedia and geeksforgeeks to read about some of the techniques like dropout etc
6. I also used LLMs like Gemini pro, chatgpt, claude for learning many concepts, understanding some code and tried understanding some papers like the attention is all you need and the one on batch normalization etc. 
