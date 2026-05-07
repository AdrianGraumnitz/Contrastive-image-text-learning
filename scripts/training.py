import torch
import torch.nn.functional as F
from scripts import prediction

def save_checkpoint(path, epoch, image_encoder, text_encoder, optimizer, vocab, losses, best_score):
    torch.save({
        "epoch": epoch,
        "image_encoder_state_dict": image_encoder.state_dict(),
        "text_encoder_state_dict": text_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "losses": losses,
        "best_score": best_score,
    }, path)

def clip_loss(image_embeddings, text_embeddings):
    logits = image_embeddings @ text_embeddings.T

    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device= logits.device)

    loss_image = F.cross_entropy(logits, labels)
    loss_text = F.cross_entropy(logits.T, labels) 

    loss = (loss_image+loss_text)/2

    return loss

import torch
import torch.nn.functional as F

def clip_loss_multicaption(image_embeddings, text_embeddings, image_ids, temperature=0.07):
    """
    image_embeddings: Tensor [batch_size, embed_dim]
    text_embeddings:  Tensor [batch_size, embed_dim]
    image_ids:        Tensor [batch_size]
                      gleiche image_id bedeutet: gehört zum gleichen Bild
    """

    logits = image_embeddings @ text_embeddings.T
    logits = logits / temperature

    # positive_mask[i, j] = True, wenn Bild i und Text j zum gleichen Bild gehören
    positive_mask = image_ids[:, None] == image_ids[None, :]

    # image -> text
    log_probs_i2t = F.log_softmax(logits, dim=1)

    loss_i2t = -(log_probs_i2t * positive_mask).sum(dim=1) / positive_mask.sum(dim=1)

    # text -> image
    log_probs_t2i = F.log_softmax(logits.T, dim=1)

    loss_t2i = -(log_probs_t2i * positive_mask.T).sum(dim=1) / positive_mask.T.sum(dim=1)

    loss = (loss_i2t.mean() + loss_t2i.mean()) / 2

    return loss

def train_eval_loop(train_dataloader,
                    test_dataloader,
                    optimizer,
                    total_epochs,
                    device,
                    image_encoder,
                    text_encoder,
                    best_model_path,
                    checkpoint_path,
                    vocab,
                    losses,
                    best_score,
                    start_epoch,):
    
    for epoch in range(start_epoch, start_epoch+total_epochs):
        total_loss=0
        for images, texts, image_ids in train_dataloader:
            images = images.to(device)
            texts = texts.to(device)
            image_ids = image_ids.to(device)
            image_embeddings = image_encoder(images)

            if texts.max().item() >= text_encoder.embedding.num_embeddings:
                print("BAD BATCH")
                print("texts min:", texts.min().item())
                print("texts max:", texts.max().item())
                print("embedding size:", text_encoder.embedding.num_embeddings)
                raise ValueError("Token id outside embedding range")

            text_embeddings =text_encoder(texts)

            loss = clip_loss_multicaption(image_embeddings=image_embeddings,
                            text_embeddings=text_embeddings,
                            image_ids=image_ids)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss/ len(train_dataloader)
        losses.append(avg_loss)
        
        print(f'Epoch: {epoch} | Avg Loss: {avg_loss:.4f}')

        image_embeddings, text_embeddings, image_ids = prediction.encode_dataset(
                                                        image_encoder=image_encoder,
                                                        text_encoder=text_encoder,
                                                        dataloader=test_dataloader,
                                                        device=device)
        logits = image_embeddings@text_embeddings.T

        r1 = prediction.recall_at_k_multicaption(logits, image_ids, k=1)
        r5 = prediction.recall_at_k_multicaption(logits, image_ids, k=5)
        r10 = prediction.recall_at_k_multicaption(logits, image_ids, k=10)
        print(f"Recall@1: {r1:.4f} | Recall@5: {r5:.4f} | Recall@10: {r10:.4f}")

        score = r5

        if score>best_score:
            best_score=score

            save_checkpoint(path = best_model_path,
                            epoch = epoch,
                            image_encoder = image_encoder,
                            text_encoder = text_encoder,
                            optimizer = optimizer,
                            vocab = vocab,
                            losses = losses,
                            best_score = best_score
                            )
            
            print(f'Saved best mode under {best_model_path}')

        if (epoch + 1)%2 == 0:
            save_checkpoint(path = checkpoint_path,
                        epoch = epoch,
                        image_encoder = image_encoder,
                        text_encoder = text_encoder,
                        optimizer = optimizer,
                        vocab = vocab,
                        losses = losses,
                        best_score = best_score
                        )
            print(f'Saved Checkpoint model under {checkpoint_path}')
    return losses