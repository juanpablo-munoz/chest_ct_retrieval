import torch

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = []
        labels = []
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            embeddings.extend(model(images))
            labels.extend(target)
        return torch.stack(embeddings), torch.tensor(labels)