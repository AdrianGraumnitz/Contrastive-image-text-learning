import torch
import torch.nn.functional as F


def encode_dataset(image_encoder,
                   text_encoder,
                   dataloader,
                   device):
    image_encoder.eval()
    text_encoder.eval()

    all_image_embeddings = []
    all_text_embeddings = []
    all_image_ids = []

    with torch.no_grad():
        for images, textes, image_ids in dataloader:
            images = images.to(device)
            textes = textes.to(device)

            image_embeddings = image_encoder(images)
            text_embeddings = text_encoder(textes)

            all_image_embeddings.append(image_embeddings.cpu())
            all_text_embeddings.append(text_embeddings.cpu())
            all_image_ids.append(image_ids.cpu())

    image_embeddings=torch.cat(all_image_embeddings, dim = 0)
    text_embeddings = torch.cat(all_text_embeddings, dim = 0)
    image_ids = torch.cat(all_image_ids, dim = 0)

    return image_embeddings, text_embeddings, image_ids

def recall_at_k_multicaption(logits, image_ids, k):
    topk = logits.topk(k, dim=1).indices

    query_image_ids = image_ids.unsqueeze(1)
    retrieved_image_ids = image_ids[topk]

    correct = (retrieved_image_ids == query_image_ids).any(dim=1)

    return correct.float().mean().item()

