import numpy as np
x_train = np.array([[8, 7], [2, 5], [5, 4], [6, 8],
                    [3, 7], [1, 3], [7, 5], [4, 3]])
y_train = np.array([1, 0, 0, 1, 1, 0, 1, 0])

x_test = np.array([[6, 6], [2, 4], [5, 7], [3, 4]])
y_test = np.array([1, 0, 1, 0])
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
    if np.mean(labels) > 0.5:
        return 1
    return 0
for k in [1, 3, 5, 7]:
    preds=[]
    for x in x_test:
        preds.append(predict(x,x_train,y_train,k))
    loss = np.mean(np.array(preds) != y_test)
    print(loss,k)