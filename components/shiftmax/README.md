# Experimental Setup:
Base FFN with each layer being:
```python
[
    nn.Linear(input_size, hidden_size),
    nn.BatchNorm1d(hidden_size),
    nn.GELU()
]
```
No dropout was used to avoid bias in the results.

The final layer of the network uses ShiftMax or Softmax respectively.

Various Loss Functions where evaluated:
```python
@staticmethod
def cross_entropy_probs(probs, targets, eps=1e-12):
    probs = torch.clamp(probs, min=eps, max=1.0-eps)
    logp = torch.log(probs)
    return F.nll_loss(logp, targets)

@staticmethod
def brier_score(probs, targets, num_classes, eps=1e-12):
    probs = torch.clamp(probs, min=eps, max=1.0-eps)
    y_onehot = F.one_hot(targets, num_classes=num_classes).float()
    return torch.mean(torch.sum((probs - y_onehot) ** 2, dim=1))

@staticmethod
def focal_loss(probs, targets, alpha=1, gamma=2.0, eps=1e-12):
    probs = torch.clamp(probs, min=eps, max=1.0-eps)
    ce_loss = -torch.log(probs)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
    focal_term = (1 - pt) ** gamma
    return torch.mean(alpha * focal_term * ce_loss.gather(1, targets.unsqueeze(1)).squeeze())

@staticmethod
def jsd_loss(probs, targets, num_classes, alpha=0.1, eps=1e-12):
    probs = torch.clamp(probs, min=eps, max=1.0-eps)
    ce_loss = F.cross_entropy(torch.log(probs), targets)
    uniform = torch.ones_like(probs) / num_classes
    m = 0.5 * (probs + uniform)
    jsd = 0.5 * F.kl_div(torch.log(probs), m, reduction='batchmean') + \ 0.5 * F.kl_div(torch.log(uniform), m, reduction='batchmean')
    return ce_loss + alpha * jsd
```

The first one of them because doesn't use cross entropy directly because Cross Entropy uses internally LogSoftmax.