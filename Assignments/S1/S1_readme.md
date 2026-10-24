# Session 1: Deep learning fundamentals

This session uses small experiments to build intuition about neural networks: what activations and depth change, how embeddings can learn from context, and how training data affects generalization.

- [Open the interactive demos](https://deeplearning-playground.netlify.app/)
- [medium artical](https://medium.com/@dhruv.vyas3690/understanding-deep-learning-through-four-interactive-experiments-940bc89fceb0)

## Experiments and takeaways

### 1. Why activations matter

- **Tried:** Compare a linear classifier with a ReLU network on two concentric rings.
- **Learned:** A linear model has a straight decision boundary, nonlinear activations let a network model curved boundaries.

### 2. What depth adds

- **Tried:** Compare one linear layer, a stack of linear layers, and a deep network with ReLU activations on the same task.
- **Learned:** Stacking linear layers alone is still a linear transformation; activations make the network more expressive.

### 3. How embeddings learn from context

- **Tried:** Train a small next-token model on a synthetic language with categories of related words.
- **Learned:** Tokens with similar contexts can develop similar embeddings, even without labels that explicitly describe similarity.

### 4. Memorization and generalization

- **Tried:** Compare training and held-out performance as the amount of training data changes.
- **Learned:** The difference between training and held-out performance helps show how well a model generalizes beyond its examples.
