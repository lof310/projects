# Loss Function
_I made a loss function that avoids the **averaging** problem of losses like Mean Absolute Error and Mean Squared Error that are used in regression tasks_

> [!NOTE]
> The internal implementation of the loss function descibed here is ommited from this public repository. A description can be found below, but the exact logic is not disclosed here.

Let $\mathbf{d} = \mathbf{x} - \mathbf{y} \in \mathbb{R}^D$ denote the error vector between the prediction and the target. Conventional regression losses such as Mean Absolute Error and the Mean Squared Error, average the per-dimension contributions uniformly. Consequently the gradient is dominated by the mean of errors, allowing a single large error to be masked by many small ones.

My loss function $\mathcal{L}(\mathbf{d})$ forces the model to focus primarily on the worst‑case dimension(s), while still maintaining a soft influence from other dimensions. This is fundamentally different from averaging losses (like L1 Loss) and is particularly effective when different output dimensions have vastly difference intrinsic "difficulty", for example smooth vs high‑frequency patterns.

In its unconstrained formulation, the loss satisfies $\lim_{\|\mathbf{d}\|_2 \to 0} \mathcal{L}(\mathbf{d}) = -\infty$. For the gradient, the direction of $\nabla_{\mathbf{d}} \mathcal{L}$ remains finite and well-defined but it's norm scales as
$$
\left\|\nabla_{\mathbf{d}} \mathcal{L}\right\|_2 \;\propto\; \frac{1}{\|\mathbf{d}\|_2},
$$

which diverges as $\|\mathbf{d}\|_2 \to 0$. In practice, numerical explosion is avoided by normalising inputs and targets to $[-1, 1]$, combined with the inherent impossibility of achieving exact zero error in real-world optimization.

I made another version whose minimum is $0$ and it's gradient is well-behaved and still have the same properties, but empirically it still outperforms L1 and L2 Losses on classification but doesn't reach the performance of the original loss function.

See: [`loss.ipynb`](loss.ipynb)
