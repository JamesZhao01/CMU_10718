# === QUERY ===

I am trying to implement a jaccard-similarity based recommender, where the
predicted rating between a user and a query item is performed through a
weighted-average of a user's consumed ratings, weighted by the
query-item-consumed-item jaccard similarity.

When i do a perfect-weighted average (e.x.
$\sum_j w(q, j) r(u, j) / \sum_j w(w,j)$, where j is the user's history), the
model's output is terribly bad (ndcg of 0.001).

However, if I make ALL the item-item jaccard similarity weights row-wise sum to
1, that is $w(q, i) = w(q, i) / \sum_{j \in \textrm{all items}} w(q, j)$) and
then do a (non-normalized) weighted average $\sum_j w(q, j) r(u, j)$ where j is
STILL the user's history (just that the weights were normalized with respect to
other, non-consumed items), the model does well (ndcg of 0.22). NOTE: This case
does NOT normalize the weights in the denominator -- the raw values are used.

Why might this be happening? The proper weighted-average does well, but the
similarity-matrix-normalized method (an improper weighted average) does poorly

# === ANSWER ===

What you're describing is an interesting issue with how your weights are
normalized in the recommendation process. Let's break it down and think about
why this might be happening.

### Method 1: Standard Weighted-Average Prediction (Low NDCG)

In your first method, you are using a standard weighted-average formula:

\[ \hat{r}(u, q) = \frac{\sum_j w(q, j) \cdot r(u, j)}{\sum_j w(q, j)} \]

Here, you're calculating the weighted sum of the user's past ratings for
consumed items \( j \), weighted by the Jaccard similarity \( w(q, j) \) between
the query item \( q \) and each item \( j \) the user has interacted with. This
approach makes intuitive sense because the predicted rating for \( q \) is a
weighted average of the user's past ratings, with more weight given to items
similar to \( q \).

However, this method performs poorly (NDCG = 0.001), suggesting that the weights
\( w(q, j) \) might not be meaningful in the sense of capturing similarity in a
way that translates well to prediction accuracy.

### Method 2: Row-wise Normalized Weights with No Denominator Normalization (High NDCG)

In the second method, you normalize the weights for each query item \( q \) by
dividing by the sum of all the weights for all items (not just the consumed
items):

\[ w'(q, j) = \frac{w(q, j)}{\sum_k w(q, k)} \]

This normalization makes the weights for each row (query item \( q \)) sum to 1.
Afterward, you predict the rating using:

\[ \hat{r}(u, q) = \sum_j w'(q, j) \cdot r(u, j) \]

Here, you're not normalizing by the sum of weights in the denominator; you're
using a straightforward dot product between the user's past ratings and these
row-normalized weights.

### Why Might This Be Happening?

1. **Smoothing Effect of Row-wise Normalization:** The row-wise normalization
   ensures that for each query item \( q \), the similarity values for all items
   sum to 1. This could have a "smoothing" effect on the weights, preventing
   overly large or small weights that could dominate the prediction in the
   standard weighted-average case.

   - **In Method 1**, if there are items \( j \) with very high or low Jaccard
     similarity values, these could either dominate the prediction (if the
     weight is large) or contribute very little (if the weight is small), making
     the final prediction noisy or biased.
   - **In Method 2**, the row-wise normalization ensures that the weights \(
     w'(q, j) \) are bounded, and this might prevent the model from assigning
     too much weight to a single item or a small subset of items.

2. **Overfitting vs Generalization:** The standard weighted-average in Method 1
   might be more prone to overfitting because the weights are directly
   proportional to the raw Jaccard similarities. If the similarities aren't
   robust enough, the predictions may be overly sensitive to specific item pairs
   that have high or low Jaccard similarity by chance, leading to poor
   generalization.

   - **Method 2** effectively smooths out the weights across all items, not just
     the user's consumed items. This "global" normalization could prevent
     overfitting to the user's specific history by considering the broader
     context of item similarities.

3. **Impact of Sparsity:** In a typical recommendation system, user-item
   interaction matrices are sparse, meaning users have interacted with only a
   small fraction of the available items.

   - **In Method 1**, you're only considering similarities with the items the
     user has consumed. If the user's history is small, this could lead to
     unstable weight values (especially if some of the \( w(q, j) \)'s are close
     to zero or vary drastically).
   - **In Method 2**, the row-wise normalization over all items might counteract
     the sparsity issue by making sure the weights are more evenly distributed,
     even if the user's history is limited.

4. **Normalization Bias:** When you row-normalize the weights, you might be
   introducing a kind of bias that favors certain items globally, which ends up
   improving the ranking of items in terms of NDCG. The Jaccard similarities
   alone might not be strong predictors, but when you normalize them with
   respect to all items, you are effectively making the query item more
   comparable to all items, not just those the user has consumed. This could
   help the model generalize better across different users and queries,
   improving NDCG.

### Why Does This Row-Wise Normalization Work Well?

- **Robustness to similarity noise:** Row-wise normalization distributes
  similarity more evenly across all items, which could prevent individual
  high-similarity items from overwhelming the recommendation.
- **Regularization effect:** It acts as a form of regularization, where extreme
  weights are avoided, preventing overfitting to specific user histories.
- **Generalization across items:** It forces the system to take into account a
  broader set of items for comparison, which might improve the quality of the
  recommendations by preventing over-reliance on potentially weak similarities.

### Conclusion:

The row-wise normalization in Method 2 helps distribute the similarity values
more uniformly, mitigating the risk of biasing predictions based on highly
similar or dissimilar items in the user's history. This "smoothing" or
regularization effect likely helps the system generalize better, which leads to
better NDCG performance, even though the method is mathematically less "correct"
from a pure weighted-average perspective.
