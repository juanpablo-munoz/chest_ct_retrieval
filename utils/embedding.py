import torch
import numpy as np

def extract_embeddings(dataloader, model):
    print("Computing embeddings...")
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        embeddings = []
        labels = []
        count = 0
        for images, target in dataloader:
            count += 1
            print(f"  Batch {count} of {len(dataloader)}")
            if torch.cuda.is_available():
                if(isinstance(images, np.ndarray)):
                    images = torch.from_numpy(images)
                else:
                    images = images.cuda()
            emb = model(images)
            embeddings.extend(emb)
            labels.extend(target)
            #print(f"Embedding shape:\n{emb.shape}\n\nTarget shape:\n{target.shape}\n------\n")
            #print(f"Embedding:\n{emb}\n\nTarget:\n{target}\n------\n")
            
            #if count >= 15:
            if count >= np.inf:
                    # end iterations early for faster debugging
                    break
        #print("\n\n--- End of embeddings calculation ---\n\n")
        #print(f"Embeddings shape: ({len(embeddings)}, {emb.shape})\n\nLabels shape: ({len(labels)}, {target.shape})\n------\n")
        #print(f"Embeddings:\n{embeddings}\n\nTargets:\n{labels}\n------\n")
        return torch.stack(embeddings), torch.tensor(labels)