import matplotlib.pyplot as plt

def similarity_matrix(logits,
                      num_images_textes,
                      ):
    sim = logits[:num_images_textes, :num_images_textes].detach().cpu()

    plt.figure(figsize = (8,8))
    plt.imshow(sim)
    plt.colorbar(label = 'Similarity')
    plt.xlabel('Text index')
    plt.ylabel('Image index')
    plt.title(f"Image-Text Similarity Matrix ({num_images_textes} samples)")
    plt.show()    

def loss_plot(losses):    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Training loss")
    plt.show()
