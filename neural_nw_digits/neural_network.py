import numpy as np
import random

class Network:
    """
    sizes: numpy array with the sizes of each layer
    [5 4 3 2], is a 4 layer nw with:
    5 neurons for layer 1
    4 neurons for layer 2
    3 neurons for layer 3
    2 neurons for layer 4

    the layer 1 is input layer, and wont have any weights or biases
    """
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights=[np.random.randn(j,k) for j,k in zip(sizes[1:],sizes[:-1])]
        self.biases =[np.random.randn(j,1) for j in sizes[1:] ]
        print(" Constructing Weights/Biases Matrix")
        for i in range(0,len(self.weights)):
            print ("=== Layer[{}]======".format(i+1))
            print(self.weights[i])

    """
    feedforward: accecpt input vector, and calculates the final output vector
    """
    def feedforward(self,a):
        # 1st a is the input, then keep updating a with new a =w*(a-1) +b
        for w,b in zip(self.weights,self.biases):
            a = sigmoid(np.dot(w,a)+b)
        return a


    def SGD(self,training_data,epochs,batch_size,eta,test_data=None):
        n_test = len(test_data)
        n_train = len(training_data)
        for epok in range(0,epochs):

            #shuffle training data for better randomness
            random.shuffle(training_data)
            #get subset of training data and submit to update_batch
            batches = [training_data[k:k+batch_size] for k in range(0,n_train,batch_size)]
            print("created [{}] batched".format(len(batches)))

            #process batches
            for batch in batches:
                self.update_weights(batch,eta)

            # evaluate against test set (with weights/biases update)
            if test_data:
                print('Epoch [{}] match/total =[{}]/[{}]]'.format(epok,
                                                     self.evaluate(test_data),
                                                     n_test)
                      )
            else:
                print("Epoch [{}] completed".format(epok))

    def update_weights(self, batch,eta):
        total_delta_w = [np.zeros(w.shape) for w in self.weights]
        total_delta_b = [np.zeros(b.shape) for b in self.biases]
        #for each tuple X,Y...get the Error
        i =0
        for (x,y) in batch:
            #print('Backpropagation batch [{}]'.format(i))
            #i+=1
            (delta_w,delta_b) = self.backpropagation(x,y)
            total_delta_w = [ tdw+dw for (dw,tdw) in zip(delta_w,total_delta_w) ]
            total_delta_b = [tdb+db for (db,tdb) in zip(delta_b, total_delta_b)]

        #upadate weights
        self.weights = [ w - (eta/len(batch))*delta_w for w,delta_w in zip (self.weights,total_delta_w)]
        self.biases = [b - (eta/len(batch))*delta_b for b, delta_b in zip(self.biases, total_delta_b)]


    # predict output and compare with test labels, return the count of match(s) between predicted and actual
    def evaluate(self, test_data):
        predicted_actual = [(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in test_data]
        boolean_vector = [a==y for (a,y) in predicted_actual]
        return sum(int(a==y) for a,y in predicted_actual)


    def backpropagation(self, x, y):
        """
        returns tuple with (nabla_W,nabla_B) representing the gradient of the cost function of X layer by layer
        nable_? is a list of numpy array of gradient of cost with same shape as weights
        nablaC_w is a list of matrix with parcial derivative of the cost function in relation to weights(jk)
        :param x:
        :param y:
        :return:
        """
        nablaC_w = [np.zeros(w.shape) for w in self.weights]
        nablaC_b = [np.zeros(b.shape) for b in self.biases]

        # need to feed forwards the X and store resuts in A[] and Z[]
        A = []
        Z = []
        # 1st activations is layer 1==input
        A.append(x)
        a = x
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,a) +b
            Z.append(z)
            a = sigmoid(z)
            A.append(a)

        #back propagate error
        # cost of last layer (a - y)
        delta_l = (A[-1] - y)*sigmoid_derivative(Z[-1])
        #same shape as self.weights[-l]
        nablaC_w[-1] = np.dot(delta_l,A[-2].transpose())
        nablaC_b[-1] = delta_l

        for l in range(2,self.num_layers):
            delta_l = np.dot(self.weights[-l+1].transpose(),delta_l)*sigmoid_derivative(Z[-l])
            nablaC_w[-l] = np.dot(delta_l, A[-l-1].transpose())
            nablaC_b[-l] = delta_l
        return nablaC_w,nablaC_b




"""
z is the vector of values.
the return is a vector as the result of applying the sigmoid function elementwise
"""
def sigmoid(z):
    return 1.0/(1.0 +np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1 - sigmoid(z))