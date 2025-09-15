'''
class machineLearner:
    def __init__(self, weights, algorithm, dataset, lossFunction):
        self.weights = weights
        self.algorithm = algorithm
        self.dataset = dataset
        self.lossFunction = lossFunction

    def learn(self):
        pass
'''
        
def predict(x, weight):
    return weight * x

def compute_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2
    #return y_true - y_pred

def adjust_weight(weight, x, y_true, y_pred, lr=0.01, eps=1e-5):
    """Approximate gradient with finite differences instead of analytic formula"""
    # Current loss
    base_loss = compute_loss(y_true, y_pred)
    
    # Slightly perturb the weight
    perturbed = weight + eps
    perturbed_pred = perturbed * x
    perturbed_loss = compute_loss(y_true, perturbed_pred)
    
    # Gradient ≈ (L(w+eps) - L(w)) / eps
    grad_approx = (perturbed_loss - base_loss) / eps
    
    return weight - lr * grad_approx
    
weight = -100
# Training data: y = 2x (we want ML to "discover" weight ~2)
#data = [(0.1, 0.2), (0.2, 0.4), (0.3, 0.6), (0.4, 0.8)]
data = [(0.1, 0.2), (0.2, 0.4), (0.3, 0.6), (0.4, 0.8)]

for epoch in range(10):  # train for 10 steps
    total_loss = 0
    for x, y_true in data:
        y_pred = predict(x, weight)
        loss = compute_loss(y_true, y_pred)
        total_loss += loss
        weight = adjust_weight(weight, x, y_true, y_pred, lr=0.01)
    print(f"Epoch {epoch+1}, Loss={total_loss:.4f}, Weight={weight:.4f}")

