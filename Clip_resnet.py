# %%
from pathlib import Path
import pandas as pd
import kagglehub
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import collections
from PIL import Image
import matplotlib.pyplot as plt
import scripts.data_preparation
import scripts.training
import scripts.models
import scripts.prediction
import scripts.plot
import scripts.datasets


# %% [markdown]
# ## Data preparation

# %% [markdown]
# ### Creating a DataFrame

# %%
path_text = Path(kagglehub.dataset_download("youssefaboelnasr/flickr8k-text"))
path_image = Path(kagglehub.dataset_download("adityajn105/flickr8k"))

df, image_dir = scripts.data_preparation.create_df(path_text=path_text,
                                        path_image=path_image)
df.head()

# %% [markdown]
# #### Check Dataintegrity

# %%
scripts.data_preparation.filter_existing_image(image_dir= image_dir,
                                               df = df)

# %%
df.shape, df['image'].nunique()

# %% [markdown]
# ### Add ImageID Column through the DataFrame

# %%
df = scripts.data_preparation.add_image_ids(df)
df.head()

# %% [markdown]
# ### Optional: Create a smaller subset for faster experiments

# %%
df = scripts.data_preparation.create_small_dataset(df,
                                                   num_images=500)

# %% [markdown]
# ### Creating a DataFrame for Train and Test

# %%
df_train, df_test, train_images, test_images = scripts.data_preparation.create_image_level_train_test_split(df = df)

# %% [markdown]
# ### Creating image preprocessing pipeline

# %%
image_size = 224

preprocess = transforms.Compose([transforms.Resize((image_size,image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                )])

# %% [markdown]
# ### Building vocabulary

# %%
vocab, counter = scripts.data_preparation.create_vocab(df_train=df_train)
vocab, counter

# %% [markdown]
# ### Creating directory for models

# %%
BATCH_SIZE = 32
checkpoint_path, best_model_path = scripts.data_preparation.create_model_directory(path = "Models/clip_bs",
                                                batch_size=BATCH_SIZE)

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% [markdown]
# ## Training

# %% [markdown]
# ### Check if a Train Checkpoint exist

# %%
if checkpoint_path.exists():
        checkpoint, vocab, image_encoder, text_encoder, optimizer, losses, best_score, start_epoch = scripts.models.load_model(checkpoint_path,
                                                                                                                               device)
        print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    
    image_encoder = scripts.models.ImageEncoder().to(device)
    text_encoder = scripts.models.TextEncoder(len(vocab)).to(device)
    image_encoder.train()
    text_encoder.train()
    optimizer = torch.optim.Adam(
    list(image_encoder.projection.parameters())+
    list(text_encoder.parameters()),
    lr = 1e-3)

    losses = []
    best_score = 0.0
    start_epoch = 0
    print("No checkpoint found. Start from scratch.")

# %% [markdown]
# ### Converting captions into token IDs

# %%
tokenizer = scripts.data_preparation.SimpleTokenizer(vocab = vocab)
tokenizer('dog , bird house cage')

# %% [markdown]
# ### Creating train- and test datasets

# %%
train_dataset = scripts.datasets.FlickrDataset(df = df_train,
                        image_dir = image_dir,
                        preprocess = preprocess,
                        tokenizer = tokenizer
                        )
test_dataset = scripts.datasets.FlickrDataset(df = df_test,
                        image_dir = image_dir,
                        preprocess = preprocess,
                        tokenizer = tokenizer
                        )

# %% [markdown]
# ### Creating train- and test dataloaders

# %%
#BATCH_SIZE = 32
NUM_WORKERS = 0

train_dataloader = DataLoader(dataset = train_dataset,
                        batch_size = BATCH_SIZE,
                        num_workers = NUM_WORKERS,
                        shuffle = True)
test_dataloader = DataLoader(dataset = test_dataset,
                        batch_size = BATCH_SIZE,
                        num_workers = NUM_WORKERS,
                        shuffle = True)

# %%
image, text, image_id = next(iter(train_dataloader))

image.shape, text.shape, image_id.shape, image_id[:10]

# %% [markdown]
# #### Debugging

# %%
batch = next(iter(train_dataloader))
images, texts, image_ids = batch

print("texts min:", texts.min().item())
print("texts max:", texts.max().item())
print("len(vocab):", len(vocab))
print("max vocab id:", max(vocab.values()))
print("embedding size:", text_encoder.embedding.num_embeddings)
print("image_ids shape:", image_ids.shape)

# %%
losses = scripts.training.train_eval_loop(train_dataloader=train_dataloader,
                                          test_dataloader=test_dataloader,
                                          optimizer=optimizer,
                                          total_epochs=3,
                                          device=device,
                                          image_encoder=image_encoder,
                                          text_encoder=text_encoder,
                                          best_model_path=best_model_path,
                                          checkpoint_path=checkpoint_path,
                                          vocab=vocab,
                                          best_score=best_score,
                                          start_epoch=start_epoch,
                                          losses=losses)

# %% [markdown]
# #### Ploting the Training loss

# %%
scripts.plot.loss_plot(losses = losses)

# %% [markdown]
# ### Prediction

# %%
if best_model_path.exists():
    best_model = torch.load(best_model_path, map_location=device)

    image_encoder.load_state_dict(checkpoint["image_encoder_state_dict"])
    text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])

    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Best score: {checkpoint['best_score']:.4f}")
else:
    print("No best model found.")

# %%
image_embeddings, text_embeddings, image_ids = scripts.prediction.encode_dataset(image_encoder=image_encoder,
                                                   text_encoder=text_encoder,
                                                   dataloader=test_dataloader,
                                                   device=device)

# %%
image_embeddings.shape, text_embeddings.shape

# %%
logits = image_embeddings@text_embeddings.T
logits.shape

# %%
for k in [1, 5, 10]:
    recall = scripts.prediction.recall_at_k_multicaption(
        logits=logits,
        image_ids=image_ids,
        k=k
    )

    print(f"Image-to-Text Recall@{k}: {recall:.4f}")

# %%
scripts.plot.similarity_matrix(logits=logits,
                  num_images_textes=50)


