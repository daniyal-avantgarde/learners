# trying to learn z = 2^x + 4^y
# loss function
# data
# predict
#
#
# i don't understand why RELU is needed
import math

wts_in2h = {
    "xh1": 0.1,
    "yh1": 0.1,
    "xh2": 0.1,
    "yh2": 0.1
}
h_nodes = { 
    "h1": None,
    "h2": None
}
wts_h2out = {
    "h1z": 0.1,
    "h2z": 0.1
}

def predict(x, y):
    # compute weighted sums into hidden layer
    # including an activation function
    h_nodes["h1"] = max(0, (
        wts_in2h["xh1"]*x + 
        wts_in2h["yh1"]*y
    ))
    h_nodes["h2"] = max(0, (
        wts_in2h["xh2"]*x + 
        wts_in2h["yh2"]*y
    ))
    
    # compute and return weighted sums into output layer
    return (
        h_nodes["h1"]*wts_h2out["h1z"] +
        h_nodes["h2"]*wts_h2out["h2z"]
    )

def backpropagation_adjust(x, y, z_true, lr=0.01):
    
    loss_gradients = {}
    
    h1 = x*wts_in2h["xh1"] + y*wts_in2h["yh1"]
    h2 = x*wts_in2h["xh2"] + y*wts_in2h["yh2"]
    
    # derivatives
    dCwrtdz = 2*(predict(x, y) - z_true)
    dzwrtwh1z = h_nodes["h1"]  # post-ReLU
    dzwrtwh2z = h_nodes["h2"]
    dzwrtdh1 = wts_h2out["h1z"]
    dzwrtdh2 = wts_h2out["h2z"]
    dh1wrtdwxh1 = x if h1 > 0 else 0
    dh2wrtdwxh2 = x if h2 > 0 else 0
    dh1wrtdwyh1 = y if h1 > 0 else 0
    dh2wrtdwyh2 = y if h2 > 0 else 0
    
    # loss gradients
    loss_gradients["lg_wh1z"] = dCwrtdz * dzwrtwh1z
    loss_gradients["lg_wh2z"] = dCwrtdz * dzwrtwh2z
    loss_gradients["lg_wxh1"] = dCwrtdz * dzwrtdh1 * dh1wrtdwxh1
    loss_gradients["lg_wxh2"] = dCwrtdz * dzwrtdh2 * dh2wrtdwxh2
    loss_gradients["lg_wyh1"] = dCwrtdz * dzwrtdh1 * dh1wrtdwyh1
    loss_gradients["lg_wyh2"] = dCwrtdz * dzwrtdh2 * dh2wrtdwyh2
    
    # update weights
    for label, loss_gradient in loss_gradients.items():
        if label[4:] in wts_in2h:
            wts_in2h[label[4:]] -= lr * loss_gradient
        elif label[4:] in wts_h2out:
            wts_h2out[label[4:]] -= lr * loss_gradient

def compute_loss(z_pred, z_true):
    return (z_pred - z_true) ** 2

data = [
    (1,1,6),
    (2,1,8),
    (1,2,18),
    (2,2,20),
    (0,1,5),
    (1,0,3),
    (3,1,12),
    #(1,3,66), this little nn literally cannot learn this datapoint
]

# implement finite differences
# then implement backprop

def finite_differences_adjust(x, y, z_true, lr=0.01, eps=1e-5):
    # disturb weight, compute loss and loss gradient, multiply by learning rate
    weights = {**wts_in2h, **wts_h2out} # ** means unpack
    loss_gradients = {}
    
    current_loss = compute_loss(predict(x,y), z_true)
    
    for label, wt in weights.items():
        
        perturbed_wt = wt + eps
        store_wt = 0
        if label in wts_in2h:
            store_wt = wts_in2h[label]
            wts_in2h[label] = perturbed_wt
        elif label in wts_h2out:
            store_wt = wts_h2out[label]
            wts_h2out[label] = perturbed_wt
        
        perturbed_loss = compute_loss(predict(x,y), z_true)
        loss_gradient = (perturbed_loss - current_loss) / eps
        loss_gradients[label] = loss_gradient
        
        if label in wts_in2h:
            wts_in2h[label] = store_wt 
        elif label in wts_h2out:
            wts_h2out[label] = store_wt
    
    for label, loss_gradient in loss_gradients.items():
        if label in wts_in2h:
            wts_in2h[label] -= loss_gradient * lr
        elif label in wts_h2out:
            wts_h2out[label] -= loss_gradient * lr
     
epochs = 20
lr = 0.01

i = 3
print(f"Datapoint: {data[i]} Predicted z: {predict(data[i][0], data[i][1]):.6f}")
print({**wts_in2h, **wts_h2out})

for epoch in range(epochs):
    total_loss = 0
    for x, y, z_true in data:
        # Forward pass
        z_pred = predict(x, y)
        # Accumulate loss
        total_loss += compute_loss(z_pred, z_true)
        # Update weights using finite differences
        #finite_differences_adjust(x, y, z_true, lr=lr)
        backpropagation_adjust(x, y, z_true, lr=lr)
    
    #print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")
print("Training ended.")

print({**wts_in2h, **wts_h2out})
print(f"Datapoint: {data[i]} Predicted z: {predict(data[i][0], data[i][1]):.6f}")