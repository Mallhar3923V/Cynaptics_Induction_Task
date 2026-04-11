
import requests
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"
def download_dataset()->None:
    if os.path.exists(DATA_PATH):
        print("File already exits. No changes made.")
        return

    text_file = requests.get(url).text
    with open(DATA_PATH,"w") as f:
        f.write(text_file)


def load_dataset(print_text = False)->str:
    
    with open(DATA_PATH, "r") as f:
        txt = f.read()

#    print("Total Characters in text: ", len(txt))
    if print_text == True:
        print(txt[:500]) 

    return txt

if __name__ == "__main__":
    download_dataset()
    load_dataset(print_text=True)
