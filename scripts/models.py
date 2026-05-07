import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        resnet = resnet18(weights = ResNet18_Weights.DEFAULT)

        for params in resnet.parameters():
            params.requires_grad = False
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(512, embed_dim)
    
    def forward(self, images):
        features = self.backbone(images)
        
        features = torch.flatten(input = features,
                                 start_dim = 1)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, dim=1)
        
        return embeddings

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim = 256, hidden_dim = 256, padding_idx = 0):
        super().__init__()

        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings = vocab_size,
                                      embedding_dim = hidden_dim,
                                      padding_idx = padding_idx)
        self.projection = nn.Sequential(
            nn.Linear(in_features= hidden_dim,
                      out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features= hidden_dim,
                      out_features = embed_dim)
                                                  )
    
    def forward(self, token_ids):
        token_embeddings = self.embedding(token_ids)
        
        mask = token_ids != self.padding_idx
        mask = mask.unsqueeze(-1)
        masked_embeddings = token_embeddings * mask
        lengths = mask.sum(dim = 1)
        lengths = lengths.clamp(min = 1) # if a word only has <tab> ->prevent 0 zero division Error
        
        text_features = masked_embeddings.sum(dim = 1)/lengths
        embeddings = self.projection(text_features)
        embeddings = F.normalize(embeddings, dim=1)
    
        return embeddings

def load_model(checkpoint_path,
               device):
    
        checkpoint = torch.load(checkpoint_path, map_location=device)
        vocab = checkpoint['vocab']

        image_encoder = ImageEncoder().to(device)
        text_encoder = TextEncoder(len(vocab)).to(device)
        image_encoder.load_state_dict(checkpoint["image_encoder_state_dict"])
        text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])

        optimizer = torch.optim.Adam(
        list(image_encoder.projection.parameters()) +
        list(text_encoder.parameters()),
        lr=1e-3
    )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        losses = checkpoint["losses"]
        best_score = checkpoint["best_score"]
        start_epoch = checkpoint["epoch"] + 1

        return (checkpoint,
                vocab,
                image_encoder,
                text_encoder,
                optimizer,
                losses,
                best_score,
                start_epoch)
  