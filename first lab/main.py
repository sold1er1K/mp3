from neural import *
from pathes import *



ROWS = 16
COLUMNS = 16
Z = 3
ERROR = 0.1

neural_network = NeuralNetwork(ROWS, COLUMNS, Z)


def main():
    finish = False
    data = prepare_data()
    while not finish:
        print('Choose a variant:')
        choice = input('1 - Learn;\n2 - Test;\n3 - Archive;\n4 - Unarchive;\n5 - Exit\n')
        if choice == 5:
            finish = True
        elif choice == 1:
            learn(neural_network, data)
        elif choice == 2:
            test(neural_network)
        elif choice == 3:
            archive(neural_network)
        elif choice == 4:
            unarchive(neural_network)
        else:
            print('!!! Wrong input !!!\n')


def prepare_data():
    count = 7
    data = numpy.zeros(((ROWS * COLUMNS // COLUMNS) * (ROWS * COLUMNS // ROWS) * count, ROWS * COLUMNS * 3))
    for i in range(count):
        pic = mpimg.imread(images_path + f"{i + 1}.png")
        for j in range(ROWS * COLUMNS // COLUMNS):
            for k in range(ROWS * COLUMNS // ROWS):
                data[ROWS * COLUMNS // ROWS * j + k] = pic[COLUMNS * j:COLUMNS * (j + 1), ROWS * k:ROWS * (k + 1), :].reshape(ROWS * COLUMNS * 3) * 2 - 1
    return data
