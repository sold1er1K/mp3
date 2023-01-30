from neural import *
from PIL import Image


data = []
for i in range(10):
    img = Image.open(f'num/{i}.png')
    num = numpy.asarray(img)
    data.append(bipolar(num.reshape(784, 1)))
    neural_network = NeuralNetwork(data)


def main():
    finish = False
    while not finish:
        print('Choose a variant:')
        choice = input('1 - Learn;\n2 - Test;\n3 - Load from file;\n4 - Exit\n')
        if choice == 4:
            finish = True
        elif choice == 1:
            neural_network.learn()
        elif choice == 2:
            number = int(input('Number Input: \n'))
            if number >= 0 and number <= 9:
                neural_network.test(number)
            else:
                print('!!! Wrong Input !!!\n')
        elif choice == 3:
            number = int(input('Number Input: \n'))
            if number >= 0 and number <= 9:
                neural_network.test_from_file(number)
            else:
                print('!!! Wrong Input !!!\n')
        else:
            print('!!! Wrong input !!!\n')

