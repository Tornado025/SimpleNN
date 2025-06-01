import numpy as np
import matplotlib.pyplot as plt     
    
class NN:

    class layer:
        def __init__(self):
            self.w=[]
            self.b=[]
            self.activation=[]
    
    def __init__(self):
        self.layers=[]
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def relu(self,z):
        return max(z,0)
    
    def sigmoid_deriv(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def relu_deriv(self,z):
        if(z>0):
            return 1
        else:
            return 0
            
    def Dense_forward(self,x,w,b,activate1):
        d=x[0].shape
        ner=np.zeros(d)
        for i in range(d):
            z=np.dot(x[:,i],w)+b[i]
            if(activate1=="relu"):
                ner[i]=self.relu(z)
            elif(activate1=="sigmoid"):
                ner[i]=self.sigmoid(z)
            else:
                ner[i]=z
        return ner
    
    def Dense_back(self,x,w,b,da,activate1,rate=0.01):
        dw=np.zeros(len(w))
        db=np.zeros(len(b))
        daprev=np.zeros(x[0].shape)
        for i in range(x[0].shape):
            z=np.dot(x[:,i],w)+b[i]
            if(activate1=="relu"):
                dz=da*self.relu_deriv(z)
            elif(activate1=="sigmoid"):
                dz=da*self.sigmoid_deriv(z)
            else:
                dz=da
            dw+=np.dot(x[i].T,dz)
            db+=dz
            daprev[i]=np.dot(w,dz)
        dw/=x[0].shape
        db/=x[0].shape
        w,b=self.update(w,b,dw,db)
        return w,b,daprev
    
    def update(w,b,dw,db,rate):
        w-=rate*dw
        b-=rate*db
        return w,b

    def train(self,x,y,w,b,activation,rate,epochs=1000):
        for i in range(epochs):
            ls=self.loss(x,y,w,b)
            for j in self.layers[::-1]:
                w,b,daprev=self.Dense_back(x,w,b,ls,activation[j],rate)

    def predict(self,x,w,b):
        d=x[0].shape
        pred=np.zeros((d,1))
        for i in range(d):
            pred[i,0]=self.forwardprop(x,w,b,"relu")
        return pred
    
    def result(self,x,w,b):
        prediction=self.predict(x,w,b)
        m=len(prediction)
        y=np.zeros(m)
        for i in range(m):
            if(prediction[i]>=0.5):
                y[i]=1
            else:
                y[i]=0
        return y
    
    def loss(self,x,y,w,b):
        pred=self.result(x,w,b)
        c=0
        for i in range(len(y)):
            if(y[i]!=pred[i]):
                c+=1
        return (c*100)/len(y)