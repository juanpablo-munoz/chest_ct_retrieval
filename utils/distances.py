import torch

def pdist(vectors):
    # distance metric: squared euclidean distance
    return torch.cdist(vectors, vectors, p=2) ** 2

def query_dataset_dist(query_tensor, dataset_tensor):
    # distance metric: squared euclidean distance
    return torch.cdist(query_tensor, dataset_tensor, p=2) ** 2

def psim_cosine(vectors):
    return torch.nn.functional.cosine_similarity(vectors[None, :, :], vectors[:, None, :], dim=-1)

def psim_cosine_query_dataset(query_tensor, dataset_tensor):
    # distance metric: 
    return -torch.nn.functional.cosine_similarity(dataset_tensor[None, :, :], query_tensor[:, None, :], dim=-1)

def pdist_dot(vectors):
    return torch.bmm(vectors[None, :, :], vectors[:, None, :].permute(1, 2, 0))
