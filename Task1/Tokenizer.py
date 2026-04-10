#implementing a custom tokenizer from hugging face
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token = "[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=12000,   
    special_tokens=["[UNK]", "<|endoftext|>"]
)

files = ["shakespeare.txt"]
tokenizer.train(files, trainer)

tokenizer.save("Shakespeare_tokenizer.json")