# 🎭 Shakespeare GPT — Decoder-Only Transformer (PyTorch)

This is my submission for the Cynaptics Club Induction Task 1.
I built a decoder-only transformer model from scratch in PyTorch to generate Shakespeare text.

---

## 🚀 How to Run

* This `Task1` folder contains all the files required to directly start training the model by running only the **GPT2.py** file.
* Only installation of the `tokenizers` library is needed:

```bash
pip install tokenizers
```

---

## 🧠 Architecture Details

* **Parameters:** ~20M parameters
  Is 20M parameters a lot? For a ~1MB dataset, it is a bit oversized, but I kept it so the model had the capacity to learn complex stylistic nuances and words that it previously did not learn on smaller `vocab_size`.

* **Layers & Heads:** 6 transformer blocks with 6 attention heads and 384 embedding dimension

* **Activation:** Used GELU instead of ReLU to avoid dead neurons

* **Optimizer:** AdamW with a learning rate of 1e-4

---

## 🔤 Tokenizer

I trained a custom Byte-Pair Encoding (BPE) tokenizer using Hugging Face's `tokenizers` library.

* Initially tried `vocab_size = 301`, but it split words into fragments like `'d es'`
* Tried `vocab_size = 5000`, which gave better results
* Finally used `vocab_size = 12000`, where complex words were captured properly without fragmentation

Final choice: **vocab_size = 12000**

---

## ⚙️ Hyperparameters

| Parameter       | Value | Notes                                    |
| --------------- | ----- | ---------------------------------------- |
| `n_embd`        | 384   | Embedding dimension per token            |
| `n_head`        | 6     | Attention heads per block (64 dims each) |
| `n_layer`       | 6     | Transformer blocks stacked               |
| `block_size`    | 256   | Maximum context length in tokens         |
| `batch_size`    | 64    | Sequences processed per training step    |
| `dropout`       | 0.3   | Reduces overfitting                      |
| `learning_rate` | 1e-4  | AdamW learning rate                      |
| `total_iters`   | 1000  | Training steps (early stopping used)     |
| `eval_interval` | 50    | Validation checks                        |

**Total parameters: ~20M**

A note on model size: with a vocab size of 12,000 and 6 layers, the parameter count is higher than typical for a dataset this small. The larger embedding dimension helps represent the wider vocabulary more expressively. The tradeoff is faster overfitting, which is mitigated using dropout (0.3) and early stopping.

---

## 🌡️ Temperature Control

I have implemented temperature-based sampling:

* Lower temperature → more repetitive, high-probability words
* Higher temperature → more creative and diverse output

---

## 📊 Sample Outputs

### Example 1

**Settings:** dropout = 0.3, temperature = 0.7, tokens generated = 256
**Input:**

```
To be or not to be
```

<details>
<summary>Click to expand full output</summary>

```
To be, or not to be born.
From my aid to go, my noble late is the day.
What is! what is my lord,
DUCHESS OF YORK:
My Lord:
At the morning?
Take your love you may be a cause.
Then did I'll have made it is the Earl of safeguard of the duke?
KING EDWARD IV:
Good cousin! what's doom?
ROMEO:
A happy day,
To complain, but a sake, thou wash him so much.
QUEEN ELIZABETH:
KING HENRY VI:
No, my lord, well;
I am too,
The king, my lord.
My lord, brother is there, like manacles, that I from the crown, good lord, I'll not of my lord.
And not,
To make a case are weak tears to me the king:
Some power.
Now never tears to the king, and divine and his intent, Aufidius,
And thou shalt not, I pray,
Or, my lord,
Or, I know I have made himself:
'er struck in this royal king.
Brother, your father would I think.
```

</details>

---

### Example 2

**Settings:** dropout = 0.4, temperature = 0.7, tokens generated = 256
**Input:**

```
To be, or not to be
```

<details>
<summary>Click to expand full output</summary>

```
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
```

</details>

---

## 📚 References and Resources

1. Andrej Karpathy — GPT-2 lecture & repo
   https://www.youtube.com/watch?v=kCc8FmEb1nY
   https://github.com/karpathy/ng-video-lecture

2. Patrick Loeber — PyTorch tutorials
   https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

3. Hugging Face Tokenizers
   https://huggingface.co/docs/tokenizers/quicktour

4. 3Blue1Brown & Jay Alammar
   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
   https://jalammar.github.io/illustrated-transformer/

5. Additional reading: Wikipedia, GeeksforGeeks

6. Used LLMs (ChatGPT, Claude, Gemini) for concept understanding and learning many concepts and also for code clarity

---
