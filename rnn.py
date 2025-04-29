import numpy as np

sentence = ["I", "study", "at", "DU"]

vocab = list(set(sentence))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

vocab_size = len(vocab)
input_size = vocab_size
hidden_size = 8
output_size = vocab_size

def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec

X = [one_hot(word_to_idx[word], vocab_size) for word in sentence[:-1]]  
Y = one_hot(word_to_idx[sentence[-1]], vocab_size)                    

np.random.seed(1)
Wxh = np.random.randn(hidden_size, input_size) * 0.01  
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  
Why = np.random.randn(output_size, hidden_size) * 0.01  

bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy(pred, target):
    return -np.sum(target * np.log(pred + 1e-9))

learning_rate = 0.1

for epoch in range(1000):
    xs, hs = {}, {}
    hs[-1] = np.zeros((hidden_size, 1))  

    for t in range(len(X)):
        xs[t] = X[t].reshape(-1, 1)
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)
    
    y_hat = np.dot(Why, hs[len(X) - 1]) + by
    probs = softmax(y_hat)
    loss = cross_entropy(probs, Y)

  
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)

    dy = probs - Y.reshape(-1, 1) 
    dWhy += np.dot(dy, hs[len(X) - 1].T)
    dby += dy

    dh = np.dot(Why.T, dy)
    
    for t in reversed(range(len(X))):
        dtanh = (1 - hs[t] ** 2) * dh
        dbh += dtanh
        dWxh += np.dot(dtanh, xs[t].T)
        dWhh += np.dot(dtanh, hs[t - 1].T)
        dh = np.dot(Whh.T, dtanh)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

   
    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

    if epoch % 100 == 0:
        pred_word = idx_to_word[np.argmax(probs)]
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Prediction: {pred_word}")


y_hat = np.dot(Why, hs[len(X) - 1]) + by
probs = softmax(y_hat)
predicted_idx = np.argmax(probs)
predicted_word = idx_to_word[predicted_idx]

print("\n main sentence: ", " ".join(sentence[:-1]))
print("target word: ", sentence[-1])
print("expected word:" , predicted_word)