import pickle, gzip, numpy as np
import random

def activate(value):
    if value > 0:
        return 1
    return 0

VECTOR_LENGTH=784

with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')
    allClassified = False

    ITERATION_NUMBER = 30
    learning_rate=0.02

    weights = np.random.randn(VECTOR_LENGTH, 10)
    biases = np.random.randn(1, 10)
    expected_outputs=np.array([1 if i == j else 0 for i in range(10) for j in range(10)]).reshape(10,10) # Matrix of 0 except when i=j the value is 0
    wrong_classified_counter=0

    while allClassified == False and ITERATION_NUMBER > 0:
        failed_classifications = [0] * 10
        allClassified = True
        data = train_set[0]                                                 #load input
        labels = train_set[1]                                               #load annotations
        permutation=[i for i in range(len(data))]
        random.shuffle(permutation) #shuffle input

        for index in permutation:
            vector = np.resize(data[index], (1, VECTOR_LENGTH))             #input vector
            v_label = labels[index]                                         #label
            z = (vector.dot(weights)+biases).flatten()                      #compute net input
            biggest=np.argmax(z)                                            #retain biggest input (in case we activate more than one perceptron)
            output = np.array([activate(z[i]) for i in range(0, 10)])       #activation vector with all perceptrons
            x = np.transpose(np.repeat(vector, 10, axis=0))
            t_minus_output = expected_outputs[v_label] - output
            weights = weights + t_minus_output * x * learning_rate          #adjust the wights
            biases = biases + t_minus_output * learning_rate                #adjust the bias

            if sum(output)!=1:                                              #If more then one perceptron is activated we can only check if the biggest input matches the expected value
                if output[biggest]!=expected_outputs[v_label][biggest]:
                    failed_classifications[v_label] += 1
                    wrong_classified_counter+=1
                    allClassified=False
            else:                                                           #If only one perceptron is activated we check if the output vector matches the expected output
                for i in range(0,10):
                    if output[i]!=expected_outputs[v_label][i]:
                        failed_classifications[v_label] += 1
                        wrong_classified_counter+=1
                        allClassified=False
                        break
        print("Wrong classified elements for iteration "+str(ITERATION_NUMBER)+":",wrong_classified_counter)
        print(failed_classifications)
        if ITERATION_NUMBER%5==0:
            learning_rate*=0.9
            print("Learning rate decreased to:",learning_rate)
        wrong_classified_counter=0
        ITERATION_NUMBER -= 1

    ITERATION_NUMBER=1
    allClassified = False


    print("TEST SET:")
    failed_classifications=[0]*10
    data = test_set[0]
    labels = test_set[1]
    for index in range(len(data)):
        vector = np.resize(data[index], (1, VECTOR_LENGTH))                 #input vector
        v_label = labels[index]                                             #label
        z = (vector.dot(weights)+biases).flatten()                          #compute net input
        biggest=np.argmax(z)                                                #retain biggest input (in case we activate more than one perceptron)
        output = np.array([activate(z[i]) for i in range(0, 10)])           #activation vector with all perceptrons

        if sum(output) != 1:                                                #If more then one perceptron is activated we can only check if the biggest input matches the expected value
            if output[biggest] != expected_outputs[v_label][biggest]:
                failed_classifications[v_label] += 1
                wrong_classified_counter += 1
                allClassified = False
        else:                                                               #If only one perceptron is activated we check if the output vector matches the expected output
            for i in range(0, 10):
                if output[i] != expected_outputs[v_label][i]:
                    failed_classifications[v_label] += 1
                    wrong_classified_counter += 1
                    allClassified = False
                    break

    print("Wrong classified elements ", wrong_classified_counter)
    print("Success ratio: "+str((len(data)-wrong_classified_counter)/100)+"%")
    print("Distribution of failed classifications:",failed_classifications)
    wrong_classified_counter=0
    ITERATION_NUMBER -= 1

    a_file = open("weights.txt", "w")
    for row in weights:
        np.savetxt(a_file, row)

    a_file.close()