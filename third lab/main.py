from neural import *
import json


def main():
    finish = False
    while not finish:
        print('Choose a variant:')
        choice = input('1 - Learn;\n2 - Test;\n3 - Exit\n')
        if choice == 3:
            finish = True
        elif choice == 1:
            seq = convert_seq(int(input('1 - Fibbonaci sequence\n2 - Periodical sequence\n3 - Factorial sequence\nPower sequence\n')))
            neural_network = NeuralNetwork(load_seq(seq))
            neural_network.learn()
        elif choice == 2:
            seq = convert_seq(int(input('1 - Fibbonaci sequence\n2 - Periodical sequence\n3 - Factorial sequence\nPower sequence\n')))
            neural_network = NeuralNetwork(load_seq(seq))
            neural_network.predict()
        else:
            print('!!! Wrong input !!!\n')


def load_seq(seq):
    with open(f'seq/{convert_seq(seq)}.json', 'r') as f:
        data = json.load(f)
    i = 1
    data = numpy.array(data[:4])
    while i < 5:
        d = numpy.append(d, d[i:i + 4])
        i = i + 1
    data = data.reshape((5, 4))
    return data


def convert_seq(num):
    if num == 1:
        return 'fib'
    elif num == 2:
        return 'period'
    elif num == 3:
        return 'fact'
    elif num == 4:
        return 'power'
