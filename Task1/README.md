# cynaptics induction task 1 - shakespeare GPT

This is my submission for the cynaptics club induction task 1 . I built a decoder-only transformer model from scratch in pytorch to generate shakespeare text 

## architecture details :
- **parameters:** around 20M parameters . Is 20M parameters a lot ? for a 1MB dataset it is a bit oversized but I kept it so the model had the capacity to learn complex stylistic nuances and words that it previously did not learn on smaller `vocab_size`
- **layers & heads:** 6 transformer blocks with 6 attention heads and 384 embedding dimension
- **activation:** used GELU intead of the standard ReLU so we don't face any issues due to dead neurons
- **Optimizer** used the AdamW , even though this optimizer requires a weight_delay parameter. I have not added it. I did implement in one of my training runs but wasn't satisfied with the output so instead of introducing another hyperparameter that I would have to tweak around and figure out the optimum value of I scrapped that idea from the final submission

## tokenizer :
I trained a custom Byte-Pair Encoding (BPE) tokenizer that I imported from the hugging face's tokeinizers library. Initially I tried a vocab_size of 301 but it kept splitting words into sub-word fragments like 'd es' . 
I also experimented with the `vocab_size` of 5000 which gave better results than 301
but lastly I increased the vocab_size to 12000 . at vocab_size of 12000 I found that it had picked complex words perfectly without fragmentation so I am keeping this `vocab_size` of 12000  

along with this I have also implemented a temperature input prompt so that the user can set temperature on their will
- lower temperature will cause the model to use words that are even slightly more likely a lot more and inturn repeat the output more
- higher temperature on the other hand will cause the model to be more creative in a sense that it will output more random and not only the more likely words
  
## Problems Faced : 
1. The problem with 12000 `vocab_size` is that the model does pick up complex words but fails to extract enough semantic/grammatical meaning from them to create a meaning ful sentence. This is becases the dataset is too small for 12000 tokenizers to learn the grammatical context in which each of them is used.
2. My model is not currently picking up the newline character after each character's name which I have to reasearch on why. I tried to briefy look into it but I am getting varied answers


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
