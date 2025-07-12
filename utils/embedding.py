import torch

def extract_embeddings(dataloader, model):
    #print("Computing embeddings...")
    with torch.autocast('cuda', dtype=torch.float16):
        with torch.no_grad():
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            embeddings = []
            labels = []
            count = 0
            for images, target in dataloader:
                count += 1
                #print(f"Batch {count} of {len(dataloader)}")
                if torch.cuda.is_available():
                    images = torch.from_numpy(images)
                    images = images.cuda()
                emb = model(images)
                embeddings.extend(emb)
                labels.extend(target)
                #print(f"Embedding:\n{emb}\n\nTarget:\n{target}\n------\n")
            return torch.stack(embeddings), torch.tensor(labels)