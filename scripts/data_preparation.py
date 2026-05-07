import pandas as pd
import numpy as np
import collections
import torch
import pathlib

class SimpleTokenizer:
    def __init__(self, vocab, max_length = 32):
        self.vocab = vocab
        self.max_length = max_length
        
    
    def __call__(self, text):
        tokens = simple_tokenize(text)

        ids = [self.vocab.get(token, self.vocab['<unk>'])
                              for token in tokens]
        ids = ids[:self.max_length]

        padding = self.max_length-len(ids)
        ids = ids + [self.vocab['<pad>']] * padding

        return torch.tensor(data = ids,
                            dtype = torch.long)

def create_df(path_text,
              path_image):
    captions_path = path_text / "Flickr8k.token.txt"
    image_dir = path_image / "Images"

    data = []

    with open(captions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            image_caption, caption = line.split("\t")
            image_name = image_caption.split(".jpg")[0] + ".jpg"
            
            data.append({
                "image": image_name,
                "caption": caption
            })

    return pd.DataFrame(data), image_dir

def filter_existing_image(image_dir, df):
    available_images = {p.name for p in image_dir.glob("*.jpg")}

    df = df[df["image"].isin(available_images)].reset_index(drop=True)

    print("Remaining rows:", len(df))
    print("Remaining unique images:", df["image"].nunique())


    print(df["image"].head(20).tolist())
    print(df[df["image"].str.contains(r"\.jpg\.\d+$", regex=True)].head())
    print("Problematische Namen:", df["image"].str.contains(r"\.jpg\.\d+$", regex=True).sum())

    df.shape, df['image'].nunique()

def add_image_ids(df):
    unique_images = df['image'].unique()
    image_name_to_id = {name: i for i, name in enumerate(unique_images)}
    df['image_id'] = df['image'].map(image_name_to_id)

    return df

def create_small_dataset(df,
                         num_images = 500,):
    
    selected_image_ids = np.random.choice(df['image_id'].unique(),
                                    size = num_images,
                                    replace = False)

    df_small= df[df['image_id'].isin(selected_image_ids)].reset_index(drop=True)
    return df_small

def create_image_level_train_test_split(df, test_size=0.2, seed=42):
    unique_images = df["image"].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_images)

    split_idx = int((1 - test_size) * len(unique_images))

    train_images = unique_images[:split_idx]
    test_images = unique_images[split_idx:]

    df_train = df[df["image"].isin(train_images)].reset_index(drop=True).copy()
    df_test = df[df["image"].isin(test_images)].reset_index(drop=True).copy()

    return df_train, df_test, train_images, test_images

def simple_tokenize(text):
        return text.lower().split()

def create_vocab(df_train):
    counter = collections.Counter()

    for caption in df_train['caption']:
        counter.update(simple_tokenize(caption))

    vocab = {
        '<pad>': 0,
        '<unk>': 1
    }

    for word,count in counter.items():
        if count >=2:
            vocab[word]= len(vocab)
    
    return vocab, counter

def create_model_directory(path,
                           batch_size):
    model_mkdir = pathlib.Path(f"{path}{batch_size}")
    model_mkdir.mkdir(parents=True,
                    exist_ok= True)

    return model_mkdir/"last_Checkpoint.pt", model_mkdir/"best_model.pt"