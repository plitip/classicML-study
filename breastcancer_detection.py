import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
 
cancer = load_breast_cancer()
n_samples = len(cancer.data)
 
labels_all = 1 - cancer.target

n, d = cancer.data.shape
print("n = ", n, "d = ", d)
print("class counts [benign(0), malignant(1)]: ", np.bincount(labels_all))
 
x_rest, X_test, y_rest, y_test = train_test_split(
    cancer.data, labels_all, test_size=0.2, random_state=0
)
 
X_train, X_val, y_train, y_val = train_test_split(
    x_rest, y_rest, test_size=0.25, random_state=0
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
 
train_preds = []
for x in X_train:
    train_preds.append(predict(x, X_train, y_train, 1))
print("k = 1 train loss: ", np.mean(np.array(train_preds) != y_train))
 
preds = []
X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])
for x in X_test:
        preds.append(predict(x,X_full,y_full,best_k))
test_loss = np.mean(np.array(preds) != y_test)
print(best_k, test_loss)
 
preds = np.array(preds)
missed_cancers = np.sum((preds == 0) & (y_test == 1))
false_alarms = np.sum((preds == 1) & (y_test == 0))
print("missed cancers: ", missed_cancers)
print("false alarms: ", false_alarms)
 