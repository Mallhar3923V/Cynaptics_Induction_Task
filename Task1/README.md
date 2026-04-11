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
  

## Sample inputs and outputs with varied inputs and outputs
- droupout = 0.3, Temperature = 0.7
  Input = To be or not to be
  Output : <br>
To be, or not to be born.
<br>
From my aid to go, my noble late is the day.
<br>
<br>
What is! what is my lord,
<br>
DUCHESS OF YORK:
My Lord:
At the morning?
<br>
Take your love you may be a cause.
<br>
<br>
Then did I'll have made it is the Earl of safeguard of the duke?
<br>
KING EDWARD IV:
Good cousin! what's doom?
<br>
ROMEO:
A happy day,
To complain, but a sake, thou wash him so much.
<br>
QUEEN ELIZABETH:
KING HENRY VI:
No, my lord, well;
<br>
I am too,
The king, my lord.
<br>
My lord, brother is there, like manacles, that I from the crown, good lord, I'll not of my lord.
<br>
And not,
<br>
To make a case are weak tears to me the king:
Some power.
<br>
<br>
Now never tears to the king, and divine and his intent, Aufidius,
And thou shalt not, I pray,
Or, my lord,
Or, I know I have made himself:
'er struck in this royal king.
<br>
<br>

Brother, your father would I think.
<br>
- dropout = 0.4, Temperature = 0.7<br>
  input : To be, or not to be<br>
  output :<br>
  To be, or not to br Murderer:
Which we have aid'd to-night;
For his day be done with a presence,
With all of Lancaster and my life shall be the harvest of this hour:
At the morning doth?
Take your enemy.


That I cannot-morrow.



I'll find him.

My lord?
BRUTUS:
And, my lord.

Nurse:
First:
ROMEO:
A happy day and let them.
LEONTES:
QUEEN ELIZABETH:
And, good; I shall be.

But that's gage.


Why,


I am too,

Where my liege, what;
And now in his life, stay not in the lost a little;
And these arms, I'll not.

When you, I not, who doth your grace,
MARCIUS:
Your lord.
I have left my knee, and I'll be tears to the king,
Yet die,
The noble lord?
RIVERS:
And yet,
The state, with the their heads of the world, I have made himself to make you know I am not to-morrow,
Is I will prove your heart would I think.
  



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
