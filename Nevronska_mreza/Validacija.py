from math import exp
from random import random, seed


# Иницијализација на мрежа
# Ставете фиксни тежини од 0.5 на code.finki.ukim.mk ако постои проблем со random()
def initialize_network(n_inputs, n_hidden, n_outputs):
    """Изградба на мрежата и иницијализација на тежините

    :param n_inputs: број на неврони во влезниот слој
    :type n_inputs: int
    :param n_hidden: број на неврони во скриениот слој
    :type n_hidden: int
    :param n_outputs: број на неврони во излезниот слој
                      (број на класи)
    :type n_outputs: int
    :return: мрежата како листа на слоеви, каде што секој
             слој е речник со клуч 'weights' и нивните вредности
    :rtype: list(list(dict(str, list)))
    """
    network = list()
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]}
                    for _ in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]}
                    for _ in range(n_outputs)]
    network.append(output_layer)
    return network


def neuron_calculate(weights, inputs):
    """Пресметување на вредноста за активација на неврон

    :param weights: даден вектор (листа) на тежини
    :type weights: list(float)
    :param inputs: даден вектор (листа) на влезови
    :type inputs: list(float)
    :return: пресметка на невронот
    :rtype: float
    """
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def sigmoid_activation(activation):
    """Sigmoid активациска функција

    :param activation: вредност за активациската функција
    :type activation: float
    :return: вредност добиена од примена на активациската
             функција
    :rtype: float
    """
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    """Пропагирање нанапред на влезот кон излезот на мрежата

    :param network: дадената мрежа
    :param row: моменталната податочна инстаца
    :return: листа на излезите од последниот слој
    """
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = neuron_calculate(neuron['weights'], inputs)
            neuron['output'] = sigmoid_activation(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def sigmoid_activation_derivative(output):
    """Пресметување на изводот на излезот од невронот

    :param output: излезни вредности
    :return: вредност на изводот
    """
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    """Пропагирање на грешката наназад и сочувување во невроните

    :param network: дадена мрежа
    :type network: list(list(dict(str, list)))
    :param expected: очекувани вредности за излезот
    :type expected: list(int)
    :return: None
    """
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_activation_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    """Ажурирање на тежините на мрежата со грешката

    :param network: дадена мрежа
    :type network: list(list(dict(str, list)))
    :param row: една инстанца на податоци
    :type row: list
    :param l_rate: рата на учење
    :type l_rate: float
    :return: None
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs, verbose=True):
    """Тренирање на мрежата за фиксен број на епохи

    :param network: дадена мрежа
    :type network: list(list(dict(str, list)))
    :param train: тренирачко множество
    :type train: list
    :param l_rate: рата на учење
    :type l_rate: float
    :param n_epoch: број на епохи
    :type n_epoch: int
    :param n_outputs: број на неврони (класи) во излезниот слој
    :type n_outputs: int
    :param verbose: True за принтање на лог, инаку False
    :type: verbose: bool
    :return: None
    """
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0] * n_outputs
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        #if verbose:
         #   print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    """Направи предвидување

    :param network: дадена мрежа
    :type network: list(list(dict(str, list)))
    :param row: една податочна инстанца
    :type row: list
    :return: предвидени класи
    """
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


dataset = [
        [6.3, 2.9, 5.6, 1.8, 0],
        [6.5, 3.0, 5.8, 2.2, 0],
        [7.6, 3.0, 6.6, 2.1, 0],
        [4.9, 2.5, 4.5, 1.7, 0],
        [7.3, 2.9, 6.3, 1.8, 0],
        [6.7, 2.5, 5.8, 1.8, 0],
        [7.2, 3.6, 6.1, 2.5, 0],
        [6.5, 3.2, 5.1, 2.0, 0],
        [6.4, 2.7, 5.3, 1.9, 0],
        [6.8, 3.0, 5.5, 2.1, 0],
        [5.7, 2.5, 5.0, 2.0, 0],
        [5.8, 2.8, 5.1, 2.4, 0],
        [6.4, 3.2, 5.3, 2.3, 0],
        [6.5, 3.0, 5.5, 1.8, 0],
        [7.7, 3.8, 6.7, 2.2, 0],
        [7.7, 2.6, 6.9, 2.3, 0],
        [6.0, 2.2, 5.0, 1.5, 0],
        [6.9, 3.2, 5.7, 2.3, 0],
        [5.6, 2.8, 4.9, 2.0, 0],
        [7.7, 2.8, 6.7, 2.0, 0],
        [6.3, 2.7, 4.9, 1.8, 0],
        [6.7, 3.3, 5.7, 2.1, 0],
        [7.2, 3.2, 6.0, 1.8, 0],
        [6.2, 2.8, 4.8, 1.8, 0],
        [6.1, 3.0, 4.9, 1.8, 0],
        [6.4, 2.8, 5.6, 2.1, 0],
        [7.2, 3.0, 5.8, 1.6, 0],
        [7.4, 2.8, 6.1, 1.9, 0],
        [7.9, 3.8, 6.4, 2.0, 0],
        [6.4, 2.8, 5.6, 2.2, 0],
        [6.3, 2.8, 5.1, 1.5, 0],
        [6.1, 2.6, 5.6, 1.4, 0],
        [7.7, 3.0, 6.1, 2.3, 0],
        [6.3, 3.4, 5.6, 2.4, 0],
        [5.1, 3.5, 1.4, 0.2, 1],
        [4.9, 3.0, 1.4, 0.2, 1],
        [4.7, 3.2, 1.3, 0.2, 1],
        [4.6, 3.1, 1.5, 0.2, 1],
        [5.0, 3.6, 1.4, 0.2, 1],
        [5.4, 3.9, 1.7, 0.4, 1],
        [4.6, 3.4, 1.4, 0.3, 1],
        [5.0, 3.4, 1.5, 0.2, 1],
        [4.4, 2.9, 1.4, 0.2, 1],
        [4.9, 3.1, 1.5, 0.1, 1],
        [5.4, 3.7, 1.5, 0.2, 1],
        [4.8, 3.4, 1.6, 0.2, 1],
        [4.8, 3.0, 1.4, 0.1, 1],
        [4.3, 3.0, 1.1, 0.1, 1],
        [5.8, 4.0, 1.2, 0.2, 1],
        [5.7, 4.4, 1.5, 0.4, 1],
        [5.4, 3.9, 1.3, 0.4, 1],
        [5.1, 3.5, 1.4, 0.3, 1],
        [5.7, 3.8, 1.7, 0.3, 1],
        [5.1, 3.8, 1.5, 0.3, 1],
        [5.4, 3.4, 1.7, 0.2, 1],
        [5.1, 3.7, 1.5, 0.4, 1],
        [4.6, 3.6, 1.0, 0.2, 1],
        [5.1, 3.3, 1.7, 0.5, 1],
        [4.8, 3.4, 1.9, 0.2, 1],
        [5.0, 3.0, 1.6, 0.2, 1],
        [5.0, 3.4, 1.6, 0.4, 1],
        [5.2, 3.5, 1.5, 0.2, 1],
        [5.2, 3.4, 1.4, 0.2, 1],
        [5.5, 2.3, 4.0, 1.3, 2],
        [6.5, 2.8, 4.6, 1.5, 2],
        [5.7, 2.8, 4.5, 1.3, 2],
        [6.3, 3.3, 4.7, 1.6, 2],
        [4.9, 2.4, 3.3, 1.0, 2],
        [6.6, 2.9, 4.6, 1.3, 2],
        [5.2, 2.7, 3.9, 1.4, 2],
        [5.0, 2.0, 3.5, 1.0, 2],
        [5.9, 3.0, 4.2, 1.5, 2],
        [6.0, 2.2, 4.0, 1.0, 2],
        [6.1, 2.9, 4.7, 1.4, 2],
        [5.6, 2.9, 3.6, 1.3, 2],
        [6.7, 3.1, 4.4, 1.4, 2],
        [5.6, 3.0, 4.5, 1.5, 2],
        [5.8, 2.7, 4.1, 1.0, 2],
        [6.2, 2.2, 4.5, 1.5, 2],
        [5.6, 2.5, 3.9, 1.1, 2],
        [5.9, 3.2, 4.8, 1.8, 2],
        [6.1, 2.8, 4.0, 1.3, 2],
        [6.3, 2.5, 4.9, 1.5, 2],
        [6.1, 2.8, 4.7, 1.2, 2],
        [6.4, 2.9, 4.3, 1.3, 2],
        [6.6, 3.0, 4.4, 1.4, 2],
        [6.8, 2.8, 4.8, 1.4, 2],
        [6.7, 3.0, 5.0, 1.7, 2],
        [6.0, 2.9, 4.5, 1.5, 2],
        [5.7, 2.6, 3.5, 1.0, 2],
        [5.5, 2.4, 3.8, 1.1, 2],
        [5.4, 3.0, 4.5, 1.5, 2],
        [6.0, 3.4, 4.5, 1.6, 2],
        [6.7, 3.1, 4.7, 1.5, 2],
        [6.3, 2.3, 4.4, 1.3, 2],
        [5.6, 3.0, 4.1, 1.3, 2],
        [5.5, 2.5, 4.0, 1.3, 2],
        [5.5, 2.6, 4.4, 1.2, 2],
        [6.1, 3.0, 4.6, 1.4, 2],
        [5.8, 2.6, 4.0, 1.2, 2],
        [5.0, 2.3, 3.3, 1.0, 2],
        [5.6, 2.7, 4.2, 1.3, 2],
        [5.7, 3.0, 4.2, 1.2, 2],
        [5.7, 2.9, 4.2, 1.3, 2],
        [6.2, 2.9, 4.3, 1.3, 2],
        [5.1, 2.5, 3.0, 1.1, 2],
        [5.7, 2.8, 4.1, 1.3, 2],
        [6.4, 3.1, 5.5, 1.8, 0],
        [6.0, 3.0, 4.8, 1.8, 0],
        [6.9, 3.1, 5.4, 2.1, 0],
        [6.8, 3.2, 5.9, 2.3, 0],
        [6.7, 3.3, 5.7, 2.5, 0],
        [6.7, 3.0, 5.2, 2.3, 0],
        [6.3, 2.5, 5.0, 1.9, 0],
        [6.5, 3.0, 5.2, 2.0, 0],
        [6.2, 3.4, 5.4, 2.3, 0],
        [4.7, 3.2, 1.6, 0.2, 1],
        [4.8, 3.1, 1.6, 0.2, 1],
        [5.4, 3.4, 1.5, 0.4, 1],
        [5.2, 4.1, 1.5, 0.1, 1],
        [5.5, 4.2, 1.4, 0.2, 1],
        [4.9, 3.1, 1.5, 0.2, 1],
        [5.0, 3.2, 1.2, 0.2, 1],
        [5.5, 3.5, 1.3, 0.2, 1],
        [4.9, 3.6, 1.4, 0.1, 1],
        [4.4, 3.0, 1.3, 0.2, 1],
        [5.1, 3.4, 1.5, 0.2, 1],
        [5.0, 3.5, 1.3, 0.3, 1],
        [4.5, 2.3, 1.3, 0.3, 1],
        [4.4, 3.2, 1.3, 0.2, 1],
        [5.0, 3.5, 1.6, 0.6, 1],
        [5.9, 3.0, 5.1, 1.8, 0],
        [5.1, 3.8, 1.9, 0.4, 1],
        [4.8, 3.0, 1.4, 0.3, 1],
        [5.1, 3.8, 1.6, 0.2, 1],
        [5.5, 2.4, 3.7, 1.0, 2],
        [5.8, 2.7, 3.9, 1.2, 2],
        [6.0, 2.7, 5.1, 1.6, 2],
        [6.7, 3.1, 5.6, 2.4, 0],
        [6.9, 3.1, 5.1, 2.3, 0],
        [5.8, 2.7, 5.1, 1.9, 0],
    ]

if __name__ == '__main__':

    seed(1)

    att1 = float(input())
    att2 = float(input())
    att3 = float(input())
    att4 = float(input())
    planttype = int(input())

    testCase = [att1,att2,att3,att4,planttype]

    learn_data = dataset[:-10]
    test_data = dataset[-10:]

    networks = [initialize_network(4,3,3) for i in range(3)]

    l_rate = 0.3
    step = 0.2

    for network in networks:
        train_network(network,learn_data,l_rate,20,3)
        l_rate += step

    maxN = -float("inf")
    index = 0
    for i in range(len(networks)):
        count = 0
        for row in test_data:
            prediction = predict(networks[i],row)
            if prediction == row[-1]:
                count += 1
        if count > maxN:
            maxN = count
            index = i

    best_network = networks[index]
    final_prediction = predict(best_network,testCase)
    print(final_prediction)