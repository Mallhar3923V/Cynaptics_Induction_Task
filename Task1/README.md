# Cynaptics induction task 1 - shakespeare GPT

This is my submission for the cynaptics club induction task 1 . I built a decoder-only transformer model from scratch in pytorch to generate shakespeare text 
## How to run : 
- This Task1 Folder contains all the files required to directly start training the model by running only the **GPT2.py** file only the installation of the tokenisers library is needed

## Architecture details :
- **parameters:** around 20M parameters . Is 20M parameters a lot ? for a 1MB dataset it is a bit oversized but I kept it so the model had the capacity to learn complex stylistic nuances and words that it previously did not learn on smaller `vocab_size`
- **layers & heads:** 6 transformer blocks with 6 attention heads and 384 embedding dimension
- **activation:** used GELU intead of the standard ReLU so we don't face any issues due to dead neurons
- **Optimizer** AdamW with a learning rate of 1e-4  

## tokenizer :
I trained a custom Byte-Pair Encoding (BPE) tokenizer that I imported from the hugging face's tokeinizers library. Initially I tried a vocab_size of 301 but it kept splitting words into sub-word fragments like 'd es' . 
I also experimented with the `vocab_size` of 5000 which gave better results than 301
but lastly I increased the vocab_size to 12000 . at vocab_size of 12000 I found that it had picked complex words perfectly without fragmentation so I am keeping this `vocab_size` of 12000  

## HyperParameters : 
 
| Parameter | Value | Notes |
|---|---|---|
| `n_embd` | 384 | Embedding dimension per token |
| `n_head` | 6 | Attention heads per block (64 dims each) |
| `n_layer` | 6 | Transformer blocks stacked |
| `block_size` | 256 | Maximum context length in tokens |
| `batch_size` | 64 | Sequences processed per training step |
| `dropout` | 0.3 | Fraction of neurons dropped during training to reduce overfitting |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `total_iters` | 1000 | Training steps — early stopping saves the best checkpoint |
| `eval_interval` | 50 | Steps between validation loss checks |
 
**Total parameters: ~20M**  

A note on model size: with a vocab size of 12,000 and 6 layers, the parameter count is higher than typical for a dataset this small. The larger embedding dimension helps the model represent the wider vocabulary more expressively — each of the 12,000 tokens gets a richer learned vector. The tradeoff is faster overfitting, which is why dropout of 0.3 and early stopping are both important here.
along with this I have also implemented a temperature input prompt so that the user can set temperature on their will
- lower temperature will cause the model to use words that are even slightly more likely a lot more and inturn repeat the output more
- higher temperature on the other hand will cause the model to be more creative in a sense that it will output more random and not only the more likely words <br>
  
## Problems Faced : 
1. The problem with 12000 `vocab_size` is that the model does pick up complex words but fails to extract enough semantic/grammatical meaning from them to create a meaningful sentence. This is becases the dataset is too small for 12000 tokenizers to learn the grammatical context in which each of them is used.
2. My model is not currently picking up the newline character after each character's name which I have to research on why. I tried to briefly look into it but I am getting varied answers

## Sample inputs and outputs with varied inputs and outputs

### 1. For everything same but dropout = 0.3 and no weight_decay, generating 100 tokens only, Temperature = 0.7
**Input Prompt** : To be or not to be  <br>
**Output** : To be , or not to be too much for the chance , The old chamber , and humbly been a better like a disease , take it . FRIAR LAURENCE : Ay , for every thing to - night ' ll make thee . DUKE VINCENTIO : I can you say Should not : O , Camillo , come to not for her son snow and my lord , this in whose rage . Boy : My lord , and , but they tell him speak The fool , will not your grace and let them , that the people , that the king , is

### 2. dropout = 0.2 and weight_decay = 0.1, generating 256 tokens, Temperature = 0.7
**Input Prompt** : To be or not to be the King <br>
**Output** : To be or not the be King of this way to not in this great pre - faced of the eagle , let ' s die in the realm be so appe gno ink ly sweet enemy . O , my kind ath aming on the manner of the death of his own , let ' d , may be with your breath , in a forehead cried ' s heaviness . GREMIO : O , And kiss , and I knew this wrong , By that you , I ' d in his sons . GREGORY : We will live , true , And Sorry take A grow as if ever most of thine wives , do not let before the heels . CLAUDIO : My lord , and imprison ' d with a man . QUEEN : Come , and drop on my mother , King Edward ' s name and then ; Good servant by the virtues over nurse , which you will not I do it not with a rebell !' ' d light , like hanging . YORK : He will be resolved the time be done , my sin ; I will be content to be yours ; But , my lord . KING EDWARD : No , I mean , To make him to all the head ? ROMEO : Where is the prince for my lord ! I do swear to his own mend it were not and a thing . CAPULET : Now , I will I have done . AUTOLYCUS : Pray ,  

### 3. dropout = 0.35 and no weight_decay, generating 256 tokens, Temperature = 0.69
**Input Prompt** : To be a KING is not to be the QUEEN <br>
**Output** : To be a KING is not to be the QUEEN MARGARET : Thou art so . ANNE : is , And set ' d . KING RICHARD II : Thy horse , my dear man ; thou hast thou not thy eyes ' s death ' d to the chair , I parted ? KING RICHARD II : I have made me , And , go , if we did go , I hard - wide to me . GLOUCESTER : O , which you have more , ' s death : I ' s death , in fit to me , and in thy death , my lord , and he hath not to the blood , And I was . WARWICK : And we ' d life , that adversaries , my wife . WARWICK : you quake , my lord , I ' ll believe thou supposed like ; I have . And I have heard off . NORTHUMBERLAND : Why , And what thou , And are in my father ' d with me again . KING HENRY VI : Are thou with a sins , I ' d my lords , My state , my lord . GLOUCESTER : Yet walk : Then , it , That did clear - fortune , that I ' ll put me , I beseech you , when my dangerous : I take my life ' d with her , my heart , I say ' st thou art thou hast thou never will not , my true , I take to your teeth .  

### 4. dropout = 0.3 and no weight_decay, generating 256 tokens, Temperature = 0.7  
**Input Prompt** :  To be or not to be <br>
**Output Prompt** : To be or not to be from his aid . I am to be is , And set ' d . First Citizen : And often , I ' ll not your own queen , that it . I have done , I had been done , I parted with a fault I have been too . Proceed : Grace to - yours . First Servant : I am hard , And that I pray you ; And brought me to - night ; but you did glad to ' s , And in the matter , I must go to seem ' d . Come , he hath not to the blood , my brother ; for your grace . JULIET : My life , that adversaries , my wife , I should you , Or , I have been in the whole mind : But ; I have . And I have heard off and tell me mock me with you , And I ' t : That I am not be again . CAPULET : A sister , thou with a very well . DUKE VINCENTIO : and great sound . O , what the duke ' s face : Then have done , That did clear - day , I ' d ? ISABELLA : This is , I beseech you to the matter , take my power to keep your favour ' d , and so : ' st thou art great fortune had it is a ' s the Antiates , when I ' d .

## References and resources used :  
1. The main structure of the code follows the code provided by **Andrej Karpathy** in his video on gpt-2 and the corresponding repo for the same video <br>
   https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4698s <br>
   https://github.com/karpathy/ng-video-lecture

2. I watched the pytorch basics tutorials from the following playlist from **Patrick Loeber** <br>
   https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

3. I used the tokenizer BPE that is provided by hugging face in their tokenizers library <br>
   https://huggingface.co/docs/tokenizers/quicktour

4. The 3B1B videos helped me get the intuition of a Transformer and also read **Jay Alammar's** "The illustrated Transformer" <br>
   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi <br>
   https://jalammar.github.io/illustrated-transformer/ <br>

5. I also used wikipedia and geeksforgeeks to read about some of the techniques like dropout etc
6. I also used LLMs like Gemini pro, chatgpt, claude for learning many concepts, understanding some code and tried understanding some papers like the attention is all you need and the one on batch normalization etc. 
