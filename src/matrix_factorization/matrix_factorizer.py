"""
    # Matrix Factorization Model

Given some training users, perform matrix factorization to yield user and item embeddings.

**With user-holdout**

- After fitting the model, fit new user embeddings while **freezing** the item embeddings

**With item-holdout**

- After fitting the model, find the rank of the held-out items

"""
class MatrixFactorizer:
    def __init__(self, output_dir: str, evaluator=None, save=True, *args, **kwargs):
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.save = True
        
    def fit(self):
        
        
