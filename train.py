# https://docs.aitextgen.io/tutorials/model-from-scratch/

from aitextgen.tokenizers import train_tokenizer
train_tokenizer(file_name)

data = TokenDataset(file_name, vocab_file=vocab_file,
                    merges_file=merges_file, block_size=32)

config = build_gpt2_config(
    vocab_size=5000, max_length=32, dropout=0.0, n_embd=256, n_layer=8, n_head=8)

ai = aitextgen(tokenizer_file=tokenizer_file, config=config)

ai.train(data, batch_size=16, num_steps=5000)

# Reloading custom model
# ai = aitextgen(model_folder="trained_model", tokenizer_file="aitextgen.tokenizer.json")
