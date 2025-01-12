import torch
import torch.nn.functional as F

from concurrent.futures import ProcessPoolExecutor, wait


# cosine similarity between the vectors of weights of the models
def cosine_similarity(model1, model2):

    # Flatten the models' parameters into a single vector
    model1_weights = torch.cat([p.view(-1) for p in model1.parameters()])
    model2_weights = torch.cat([p.view(-1) for p in model2.parameters()])

    # Compute the cosine similarity
    cos_sim = F.cosine_similarity(model1_weights, model2_weights, dim=0)

    return cos_sim.item()


# pool of workers for multi-process training
class Pool:

    def __init__(self, max_workers):
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.futures = []


    def submit_task(self, func, *args, **kwargs):
        future = self.executor.submit(func, *args, **kwargs)
        self.futures.append(future)


    def collect_results(self):

        # Ensure all futures are completed
        wait(self.futures)

        # Collect results from the completed futures in the order they were submitted
        results = [future.result() for future in self.futures]

        # Reset futures list for future tasks
        self.futures = []
        return results


    def shutdown(self):
        self.executor.shutdown(wait=True)