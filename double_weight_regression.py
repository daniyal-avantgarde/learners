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
def predict(x, weights):
    return weights[0] * weights[1] * x

def compute_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def adjust_weights(weights, x, y_true, y_pred, lr=0.01, delta=1e-5):
    current_loss = compute_loss(y_true, y_pred)
    loss_gradients = {}

    for i in range(len(weights)):
        perturbed_weights = weights.copy()
        perturbed_weights[i] += delta
        perturbed_pred = predict(x, perturbed_weights)
        perturbed_loss = compute_loss(y_true, perturbed_pred)
        
        loss_gradient = (perturbed_loss - current_loss) / delta
        loss_gradients[i] = loss_gradient
    
    return [weights[i] - lr * loss_gradients[i] for i in range(len(weights))]

weights = [-0.3, -2]
data = [(1, 2), (2, 4), (3, 6), (4, 8)]

for epoch in range(10):
    for x, y_true in data:
        y_pred = predict(x, weights)
        weights = adjust_weights(weights, x, y_true, y_pred, lr=0.01)
    print(f"Epoch {epoch+1}, Loss={compute_loss(y_true, y_pred):.4f}, Weights={weights}")
