import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import streamlit.components.v1 as components
#st.beta_set_page_config(layout="wide")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.title("Lottery Ticket Hypothesis ")
st.markdown("[Frankle & Carbin (2019) — The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf)")
st.markdown("""<pre>A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the
original network after training for at most the same number of iterations.</pre>
""", unsafe_allow_html=True)
st.subheader("digits dataset")


INIT_WEIGHTS="rand" #can be rand , const , scaled rand
RANDOM_SEED=0
INIT_VALUE=1 # used if INIT_WEIGHTS=="const" or "scaled_rand"
TEST_SIZE=0.1
EPOCHES=st.number_input('TOTAL EPOCHES', value=50,min_value=1,max_value=1000)

### custom nn start
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as dc

def np_random_randn(dim1,dim2):
    if(INIT_WEIGHTS=="rand"):
        np.random.seed(RANDOM_SEED)
        return np.random.randn(dim1,dim2) #mean 0 variance 1
    if(INIT_WEIGHTS=="const"):
        return np.ones((dim1, dim2))*INIT_VALUE
    if(INIT_WEIGHTS=="scaled_rand"):
        np.random.seed(RANDOM_SEED)
        return np.random.randn(dim1,dim2)*INIT_VALUE

dig = load_digits()
onehot_target = pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=0)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

class MyNN:
    def __init__(self, x, y,lr):
        self.x = x
        neurons = 4
        self.neurons=neurons
        self.lr = lr
        ip_dim = x.shape[1]
        op_dim = y.shape[1]
        #self.w1 = -np.ones((ip_dim, neurons))/2#np.random.randn(ip_dim, neurons)
        self.w1 = np_random_randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        #self.w2 = -np.ones((neurons, neurons))/2#np.random.randn(neurons, neurons)
        self.w2 = np_random_randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        #self.w3 = -np.ones((neurons, op_dim))/2#np.random.randn(neurons, op_dim)
        self.w3 = np_random_randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        self.y = y
        self.init_weights=(dc(self.w1),dc(self.w2),dc(self.w3))
        self.hist_error=[]
        self.hist_weights=[[0,],[0,],[0,]]
        self.epoches=0
        self.lottery_drawn=False
    def setweights(self,weights):
        if(self.lottery_drawn):
          raise ValueError("INITIALIZE MODEL BEFORE SETTING CUSTOM WEIGHTS ")
        self.lottery_drawn=True
        neurons = self.neurons
        ip_dim = self.x.shape[1]
        op_dim = self.y.shape[1]

        self.w1 = weights[0]
        self.b1 = np.zeros((1, neurons))
        self.w2 = weights[1]
        self.b2 = np.zeros((1, neurons))
        self.w3 = weights[2]
        self.b3 = np.zeros((1, op_dim))
    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

    def backprop(self):
        hist_weights=self.hist_weights
        hist_weights[0].append(abs(self.w1).sum())
        hist_weights[1].append(abs(self.w2).sum())
        hist_weights[2].append(abs(self.w3).sum())


        loss = error(self.a3, self.y)
        self.hist_error.append(loss)
        self.epoches+=1
        #print('Loss :', loss)
        a3_delta = cross_entropy(self.a3, self.y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)
        self.end_weights=(self.w1,self.w2,self.w3)
    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()
    def show_weights_diff(self):
        print("difference in initial and final weights : ")
        count=1
        for new,old in zip(self.end_weights,self.init_weights):
          plt.figure(figsize=(1,1))
          plt.imshow((new-old), cmap='PiYG', interpolation='nearest')
          plt.title("Weight matrix :"+str(count))
          plt.show()
          st.pyplot(plt)
          print("max diff :",(new-old).max())
          print("min diff :",(new-old).min())
          count+=1
        plt.colorbar()

    def show_weights(self,weights_array):
        print("Given  Weights are : ")
        count=1
        for new in weights_array:
          plt.figure(figsize=(1,1))
          plt.figure(15)
          plt.imshow(new, cmap='PiYG', interpolation='nearest')
          #plt.colorbar()
          plt.title("Weight matrix :"+str(count))
          count+=1
          plt.show()
          st.pyplot(plt)
          print("max weight :",(new).max())
          print("min weight :",(new).min())

    def show_loss_graph(self):
        plt.figure(19)
        plt.plot(range(0,self.epoches),self.hist_error)
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.show()
        st.pyplot(plt)
        for i in range(0,3):
          plt.figure(12)
          plt.plot(range(0,self.epoches),self.hist_weights[i][1:])
          plt.xlabel("weight"+str(i))
          plt.ylabel("value")
          plt.show()


    def train_model(self,epochs):
        for epoch in range(epochs):
          self.feedforward()
          self.backprop()

#outside class
def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100
model = MyNN(x_train/16.0, np.array(y_train),lr=10)

epochs = EPOCHES
model.train_model(epochs)
oldtest_acc=get_acc(x_val/16, np.array(y_val))
oldtrain_acc=get_acc(x_train/16, np.array(y_train))
st.text("Test accuracy : "+ str(get_acc(x_val/16, np.array(y_val)) ))
st.text("Train accuracy : "+ str(get_acc(x_train/16, np.array(y_train)) ))
#model.show_loss_graph()
model.show_weights_diff()


#### custom nn end




### lottery ticket
def get_lottery_ticket(initw,endw,threshold=0.3,mid=False,lower=False,upper=False):
  newinitw=[]
  nonzeroval=0
  totalval=0
  for i,e in zip(initw,endw):
                                                                            #normalize to -1 to 1 by min max scalar
    mat=e/(e.max()-e.min())
                                                                            #put initial weights beyond threshold
    if(type(mid)==type(123)):
      mat=np.where(np.logical_and((-threshold<mat),(mat<threshold)),mid,i)
    if(type(lower)==type(123)):
      mat=np.where(-threshold>=mat,lower,mat)
    if(type(upper)==type(123)):
      mat=np.where(threshold<=mat,upper,mat)
    newinitw.append(mat)
                                                                             #to get percent of alive weights
    nonzeroval+=np.count_nonzero(mat)
    totalval+=mat.shape[0]*mat.shape[1]

  print("neuron weights alive percent is ",nonzeroval*100/totalval)
  return newinitw
THRESHOLD=st.slider("enter a threshold ",0.1,1.0,step=0.05,value=0.3)
VALUES_NEAR_0=0

newweights=get_lottery_ticket(model.init_weights,model.end_weights,THRESHOLD,VALUES_NEAR_0)

###


###pruned model
weights=dc(newweights)
model = MyNN(x_train/16.0, np.array(y_train),1)
model.setweights(weights)
st.title("weights after applying mask")
model.show_weights(weights)

epochs = EPOCHES
model.train_model(epochs)
st.title("weight diff after training again")

model.show_weights_diff()
model.show_loss_graph()
st.text("old Test accuracy : "+ str(oldtest_acc))
st.text("old Train accuracy : "+ str((oldtrain_acc)))

st.text("new Test accuracy : "+ str(get_acc(x_val/16, np.array(y_val)) ))
st.text("new Train accuracy : "+ str(get_acc(x_train/16, np.array(y_train)) ))

###


st.markdown(hide_streamlit_style, unsafe_allow_html=True)
components.html(
    """
<script>
window.addEventListener("load", function(){
var array = [];
var links = parent.document.getElementsByTagName("a");
for(var i=0, max=links.length; i<max; i++) {
    array.push(links[i].href);
	links[i].html="hi";
}
console.log(array)
var links = parent.document.getElementsByTagName("summary");
for(var i=0, max=links.length; i<max; i++) {
    links[i].html="hi";
}

});

</script>
    """,
)
