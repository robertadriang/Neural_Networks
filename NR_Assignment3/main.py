import pickle, gzip, numpy as np
import random
# Extra documentation
# https://datascience.stackexchange.com/questions/30676/role-derivative-of-sigmoid-function-in-neural-networks - LIFESAVER :)
# 3Blue1Brown https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
# StatQuest https://www.youtube.com/c/joshstarmer

# Checklist:
#● The neural network needs to be able to obtain at least 95% accuracy on a test set - Done
#● The used cost function must be cross-entropy - Done
#● The last layer has to use the softmax activation function - Done
#● Weights must be initialized in a proper manner to avoid saturation of the neurons. - Done
#● You use at least one of the following techniques:
#o L2 regularization + momentum - Done
#o Dropout + any data augmentation
#o RMSProp

def print_image(input):
    first_image = input[0]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.title(f"Label is {input[1]} but was recognized as {input[2]}")
    plt.imshow(pixels, cmap='gray')
    plt.show()


def sigmoid(z):
    #Sigmoid function slide 7:
    #https://docs.google.com/presentation/d/1dPXMnLI4Gy8lQRilgLik1YZMcanO2JWifGHlbiFSunA/edit#slide=id.p7
    return 1.0 / (1 + np.exp(-z))


def softmax(z):
    #Softmax function slides 23-28
    #https://docs.google.com/presentation/d/1oS2f1p-_heuSsOXUU1JZKt7mSdrnE4ZN_Vbkik4dgg4
    return np.exp(z) / np.sum(np.exp(z))


def sigmoid_derivative(z):
    # https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_size = input_layer_size
        self.hidden_size = hidden_layer_size
        self.output_size = output_layer_size

        self.weights = []
        # first element Weights for hidden layer
        # 2nd element Weights for output layer
        #https://docs.google.com/presentation/d/1oS2f1p-_heuSsOXUU1JZKt7mSdrnE4ZN_Vbkik4dgg4/edit#slide=id.p40
        # (Weights initialized with a standard deviation of 1/sqrt(size of connections)
        self.weights.append(np.random.randn(self.hidden_size, self.input_size) / np.sqrt(self.input_size)) # 100 784
        self.weights.append(np.random.randn(self.output_size, self.hidden_size) / np.sqrt(self.hidden_size)) # 10 100
        # for 3 layers previous method
        # self.hidden_weights=np.random.randn(self.hidden_size,self.input_size)/np.sqrt(self.input_size)
        # self.output_weights=np.random.randn(self.output_size,self.hidden_size)/np.sqrt(self.hidden_size)
        # print("H:",self.hidden_weights.shape)
        # print("H:", len(self.hidden_weights))
        # print("H:", len(self.hidden_weights[0]))
        # print("O:",self.output_weights.shape)
        # print("O:", len(self.output_weights))
        # print("O:", len(self.output_weights[0]))
        self.biases = []
        self.biases.append(np.random.rand(self.hidden_size, 1))
        self.biases.append(np.random.rand(self.output_size, 1))
        # self.hidden_biases=np.random.rand(self.hidden_size,1)
        # self.output_biases=np.random.rand(self.output_size,1)
        # print(self.hidden_biases)
        # print(self.output_biases)

    def train(self, train_set, valid_set, test_set, iterations_number, batch_size, learning_rate,momentum_coefficient,l2_constant):
        data = train_set[0]  # load input
        labels = train_set[1]  # load annotations
        data_len = len(data)
        batches_number = data_len // batch_size
        # L2 regularization
        # https://www.analyticssteps.com/blogs/l2-and-l1-regularization-machine-learning
        # https://docs.google.com/presentation/d/1-6JxfxZLuM2LERppuTzYufzBbaNobpUh0iivLt08bTA/present?slide=id.p24
        l2_term = 2 * l2_constant/data_len

        for it_counter in range(iterations_number):
            permutation = [i for i in range(data_len)]
            random.shuffle(permutation)

            for i in range(batches_number):
                if (i+1)%1000==0:
                    print(f"Finished processsing {(i+1)*batch_size} elements in training for iteration {it_counter+1}.")
                batch_weights_update = [np.zeros(w.shape) for w in self.weights]
                batch_biases_update = [np.zeros(b.shape) for b in self.biases]
                # batch_h_w_update=np.zeros(self.hidden_weights.shape)
                # batch_o_w_update=np.zeros(self.output_weights.shape)
                # batch_h_b_update=np.zeros(self.hidden_biases.shape)
                # batch_o_b_update=np.zeros(self.output_biases.shape)
                # print(batch_h_w_update)
                # print(batch_o_w_update)
                # print(batch_h_b_update)
                # print(batch_o_b_update)
                for j in range(batch_size):
                    index=permutation[i * batch_size + j]
                    element = data[index].reshape(len(data[index]),1)
                    annotation = labels[index]

                    layers_sums, layers_activations = self.train_feedforward(element)
                    weights_adjustments, biases_adjustments = self.train_backpropagation(element, annotation,
                                                                                         layers_sums,
                                                                                         layers_activations)

                    for k in range(len(weights_adjustments)):
                        batch_weights_update[k] += weights_adjustments[k]
                        batch_biases_update[k] += biases_adjustments[k]

                friction_weights = [np.ones(w.shape) for w in self.weights]
                friction_biases = [np.ones(b.shape) for b in self.biases]
                for k in range(len(batch_weights_update)):
                    #Momentum
                    #https://docs.google.com/presentation/d/1-6JxfxZLuM2LERppuTzYufzBbaNobpUh0iivLt08bTA/edit#slide=id.p12
                    friction_weights[k] = momentum_coefficient * friction_weights[k] - learning_rate * batch_weights_update[k]
                    friction_biases = momentum_coefficient * friction_biases[k] - learning_rate * batch_biases_update[k]

                    # friction_weights[k] = momentum_coefficient * batch_weights_update[k] - learning_rate * \
                    #                       batch_weights_update[k] #???????
                    # friction_biases = momentum_coefficient * batch_biases_update[k] - learning_rate * \
                    #                   batch_biases_update[k]   #?????????

                    #Adjust weights and biases
                    self.weights[k] = self.weights[k] + friction_weights[k]-learning_rate*l2_term*self.weights[k]
                    self.biases[k] = self.biases[k] + friction_biases[k]-learning_rate*l2_term*self.biases[k]

            print("Finished iteration:", it_counter+1)

            self.test(test_set)

    def train_feedforward(self, element):
        #Feedforward algorithm: slides 3-8:
        #https://docs.google.com/presentation/d/1dPXMnLI4Gy8lQRilgLik1YZMcanO2JWifGHlbiFSunA
        previous_layer_output=element # 784,1 = 100,1
        layers_activations = [previous_layer_output]
        layers_sums = []

        for i in range(len(self.weights)):
            current_layer_sum=np.dot(self.weights[i],previous_layer_output)+self.biases[i]
            layers_sums.append(current_layer_sum)
            if i+1 != len(self.weights):
                previous_layer_output=sigmoid(current_layer_sum)
            else:
                previous_layer_output=softmax(current_layer_sum)
            layers_activations.append(previous_layer_output)
        # Method for only 3 layers (Input, Hidden, Output)
        # h_layer_sum = np.dot(self.weights[0], element) + self.biases[0]
        # layers_sums.append(h_layer_sum)
        # h_layer_output = sigmoid(h_layer_sum)
        # layers_activations.append(h_layer_output)
        # o_layer_sum = np.dot(self.weights[1], h_layer_output) + self.biases[1]
        # layers_sums.append(o_layer_sum)
        # o_layer_output = softmax(o_layer_sum)
        # layers_activations.append(o_layer_output)
        return [layers_sums, layers_activations]

    def train_backpropagation(self, element, annotation, layers_sums, layers_activations):
        #Backpropagation algorithm: slide 34
        #https://docs.google.com/presentation/d/1dPXMnLI4Gy8lQRilgLik1YZMcanO2JWifGHlbiFSunA/edit#slide=id.p34

        weights_adjustments = [np.zeros(w.shape) for w in self.weights]
        biases_adjustments = [np.zeros(b.shape) for b in self.biases]

        #Cross entropy: slide 15 (9-15 for generalized method)
        #https://docs.google.com/presentation/d/1oS2f1p-_heuSsOXUU1JZKt7mSdrnE4ZN_Vbkik4dgg4/edit#slide=id.p15
        delta = layers_activations[-1] - IDENTITY_MATRIX[annotation].reshape(len(IDENTITY_MATRIX), 1)
        weights_adjustments[-1] = np.dot(delta, layers_activations[-2].transpose())
        biases_adjustments[-1] = delta

        for i in range(2, len(self.weights) + 1):
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sigmoid_derivative(layers_sums[-i]) #Error in the previous layer
            weights_adjustments[-i] = np.dot(delta, layers_activations[-i - 1].transpose()) #Gradient for the weights in the current layer
            biases_adjustments[-i] = delta #Gradient for the biases in the current layer

        return [weights_adjustments, biases_adjustments]

    def test_feedforward(self, element):
        h_layer_sum = np.dot(self.weights[0], element) + self.biases[0]
        h_layer_output = sigmoid(h_layer_sum)
        o_layer_sum = np.dot(self.weights[1], h_layer_output) + self.biases[1]
        o_layer_output = softmax(o_layer_sum)
        return o_layer_output

    def test(self, test_set):
        wrong_classified_counter = 0
        failed_classifications = [0] * 10
        data = test_set[0]  # load input
        labels = test_set[1]  # load annotations
        for i in range(len(data)):
            element = data[i].reshape(len(data[i]),1)
            output_layer = self.test_feedforward(element)
            biggest = np.argmax(output_layer)
            if IDENTITY_MATRIX[labels[i]][biggest]!=1:
                failed_classifications[labels[i]] += 1
                #print_image((element,labels[i],biggest))
                wrong_classified_counter += 1
        print("Results:")
        print("Accuracy:",round(100*(len(data)-wrong_classified_counter)/len(data),2),"%")
        print("Distribution of unrecognized elements:",failed_classifications)


with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

IDENTITY_MATRIX = np.array([1 if i == j else 0 for i in range(10) for j in range(10)]).reshape(10,10)  # Matrix of 0 except when i=j the value is 0
ITERATIONS = 30
LEARNING_RATE = 0.02
MINI_BATCH_SIZE = 10
MOMENTUM_COEFFICIENT=0.000005
L2_CONSTANT=2

from matplotlib import pyplot as plt


mnist_classifier = NeuralNetwork(len(train_set[0][0]), 100, 10)
mnist_classifier.train(train_set, valid_set, test_set, ITERATIONS, MINI_BATCH_SIZE, LEARNING_RATE,MOMENTUM_COEFFICIENT,L2_CONSTANT)
