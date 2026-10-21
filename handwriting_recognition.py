import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
tra = np.loadtxt("optdigits.tra", delimiter=",", dtype=int)

x_rest, X_test, y_rest, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    x_rest, y_rest, test_size=0.25, shuffle=True
)

def  distance(a,b):
    dist = np.sqrt((np.sum((a-b)**2)))
    return dist
def predict(x,x_train,y_train,k):
    distances=[]
    for row in x_train:
        distances.append(distance(x,row))
    distances= np.argsort(distances)
    nearest = distances[:k]
    labels = y_train[nearest]
    
    return np.argmax(np.bincount(labels))
best_loss=500
for k in range(1,15):
    preds=[]
    for x in X_val:
        preds.append(predict(x,X_train,y_train,k))
    loss = np.mean(np.array(preds) != y_val)
    if best_loss>loss:
        best_k = k
        best_loss =loss

preds = []
X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])
for x in X_test:
        preds.append(predict(x,X_full,y_full,best_k))
test_loss = np.mean(np.array(preds) != y_test)
print(best_k, test_loss)


preds = []
X_new = tra[:, :64]   
y_new = tra[:, 64] 
for x in X_new:
    preds.append(predict(x,data ,digits.target,best_k))
test_loss2 = np.mean(np.array(preds) != y_new)
print(test_loss2)