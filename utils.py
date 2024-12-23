import torch
import torch.nn.functional as F

def cosine_similarity(model1, model2):

    # Flatten the models' parameters into a single vector
    model1_weights = torch.cat([p.view(-1) for p in model1.parameters()])
    model2_weights = torch.cat([p.view(-1) for p in model2.parameters()])

    # Compute the cosine similarity
    cos_sim = F.cosine_similarity(model1_weights, model2_weights, dim=0)

    return cos_sim.item()