# KL Divergebce
### Concept
KL divergence is a method for measuring the difference between two probability distributions.Given two probability distributions P and Q, KL divergence expresses the difficulty of representing the actual distribution using the distribution Q under the probability distribution P.

### Formulation
__KL(P || Q) = Σ P(x) * log(P(x) / Q(x))__ , where P(x) and Q(x) represent the probabilities of events x under the probability distributions P and Q, respectively.

### Features
* KL divergence is asymmetric: that is, KL(P || Q) ≠ KL(Q || P). This means that the difference between P and Q is directional.(| KL(P || Q) |= | KL(Q || P) |)
* KL divergence is non-negative: KL divergence is never less than zero, and it is zero only when P and Q are exactly the same.
* KL divergence is not a distance metric: it does not satisfy the triangle inequality, and thus does not have the properties of a metric.

In conclusion, KL divergence has wide applications in information theory, statistics, and machine learning. It can help compare and quantify the difference between two probability distributions, optimize models, and improve application effectiveness.

### Examples
```python
import numpy as np  

# 定义两个离散型概率分布P和Q  
P = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])  # 概率分布P，表示骰子的6个面的概率 
Q = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])  # 概率分布Q,另一个概率分布 

# 计算KL散度  
kl_divergence = np.sum(P * np.log(P / Q))  

print("KL散度值为:", kl_divergence)
```