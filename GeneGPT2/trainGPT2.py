
import numpy as np
import pandas as pd
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import build_gpt2_config
from aitextgen import aitextgen

print("Starting")

# Train GPT2 & Tokenizer
file_name = "./data/trainValidCorpus.txt"
train_tokenizer(file_name)
config = build_gpt2_config(
    vocab_size=1000, max_length=64, dropout=0.0, n_embd=32, n_layer=8, n_head=8)
ai = aitextgen(config=config,
               tokenizer_file="aitextgen.tokenizer.json",
               to_gpu=True)

print("Starting to train GPT2...")
ai.train(file_name,
         line_by_line=False,
         from_cache=False,
         num_steps=1000,
         generate_every=1000,
         save_every=1000,
         save_gdrive=False,
         learning_rate=1e-3,
         batch_size=64,
         )
print("GPT2 trained")

# Retrieve embeddings
word_embeddings = ai.model.transformer.wte.weight
position_embeddings = ai.model.transformer.wpe.weight

# Generate embeddings
all_prompts = []
dataPaths = ["C1Test.csv", "C1Train.csv", "C1Valid.csv", "C2Test.csv",
             "C2Train.csv", "C2Valid.csv", "DiffTrain.csv", "DiffValid.csv", "DiffTest.csv"]
for path in dataPaths:
    all_prompts.append(pd.read_csv(
        "./data/" + path, delimiter='\t', header=None))

print("Starting to generate embeddings")
for i in range(len(all_prompts)):
    embedded = []
    for prompt in all_prompts[i][0]:
        tokens = ai.tokenizer(text=prompt, return_tensors="pt")[
            "input_ids"][0].numpy()[0:207]
        w_embedding = []
        for token in tokens:
            w_embedding.append(word_embeddings[token].cpu().detach().numpy())
        w_embedding = np.asarray(w_embedding).flatten()

        embedded.append(w_embedding)
    embedded = np.asarray(embedded)

    np.savetxt("./embeddings/" + dataPaths[i][:-4] + "Embeddings.csv",
               embedded,
               delimiter=", ",
               fmt='% s')
print("Embeddings generated. trainGPT2 complete.")
