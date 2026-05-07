from PIL import Image
from torch.utils.data import Dataset

class FlickrDataset(Dataset):
    def __init__(self, df, image_dir, preprocess, tokenizer):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_dir / row["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)

        caption = row["caption"]
        text = self.tokenizer(caption)

        image_id = int(row["image_id"])

        return image, text, image_id