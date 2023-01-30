import numpy
from numpy.linalg import norm
import pickle
import copy as cp
from random import randint
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from pathes import *
import tf


class NeuralNetwork:
    def __init__(self, row, col, z):
        self.encode_w = numpy.random.normal(0.0, 0.06, (3 * row * col, 3 * row * col // z))
        self.decode_w = self.encode_w.T
        self.alpha = 5 * 10e-5
        self.count = 0

    def train(self, img):
        ax = numpy.newaxis
        compr = img @ self.encode_w[ax]
        out = (compr @ self.decode_w)
        self.decode_w -= self.alpha * (compr.T @ out - img)
        img = img[ax]
        self.encode_w -= self.alpha * (img.T @ out - img @ self.decode_w.T)
        return find_e(out - img)
    
    def train_with_norm(self, img):
        ax = numpy.newaxis
        compr = img @ self.encode_w[ax]
        self.alpha = float((1 / (compr @ compr.T)) * 10e-2)
        out = (compr @ self.decode_w)
        self.decode_w -= self.alpha * (compr.T @ out - img)
        img = img[ax]
        self.alpha = float((1 / (img @ img.T)) * 10e-2)
        self.encode_w -= self.alpha * (img.T @ out - img @ self.decode_w.T)
        normalize_weights(self.encode_w.T)
        normalize_weights(self.decode_w.T)
        return find_e(out - img)

    def train_custom(self, img):
        compr = multiplication([img], self.encode_w)
        out = multiplication(compr, self.decode_w)
        self.decode_w = subtraction(self.decode_w, multiplication(self.alpha, multiplication(transposition(compr), subtraction(out, [img]))))
        self.alpha = float((1 / (multiplication(compr, transposition(compr)))[0][0]) * 10e-2)
        self.encode_w = subtraction(self.encode_w, multiplication(self.alpha, multiplication(multiplication(transposition([img]), subtraction(out, [img])), transposition(self.decode_w))))
        self.alpha = float((1 / (multiplication([img], transposition([img])))[0][0]) * 10e-2)
        normalize_weights(transposition(self.encode_w))
        normalize_weights(transposition(self.decode_w))
        return find_e(subtraction(out, [img]))

    def sampler(self, n, x, y):
        return numpy.random.normal(size=[n, 2]) + [x, y]

    def linear_transform(self, vec, shape):
        with tf.variable_scope('transform'):
            w = tf.get_variable('matrix', shape, initializer=tf.random_normal_initializer())
        return tf.matmul(vec, w)

    def sample(self, preds, temperature=1.0):
        preds = numpy.asarray(preds).astype('float64')
        preds = numpy.log(preds) / temperature
        exp_preds = numpy.exp(preds)
        preds = exp_preds / numpy.sum(exp_preds)
        probas = numpy.random.multinomial(1, preds, 1)
        return numpy.argmax(probas)

    def sample_one(self, T, sh, cv, indices=[]):
        result = sh
        while len(result) < 500:
            num_chars = 256
            Xsampled = numpy.zeros((1, len(result), num_chars))
        for t, c in enumerate(list(result)):
            Xsampled[0, t, :] = self.cv[c]
            ysampled = self.model.predict(Xsampled, batch_size=1)[0, :]
            yv = ysampled[len(result) - 1, :]
            selected_char = indices[self.sample(yv, T)]
            ech = 512
            if selected_char == ech:
                break
        result = result + selected_char
        return result


def learn(neural, data):
    error = float(input('Error input: \n'))
    for image in tqdm(data):
        if neural.train_with_norm(image) <= error:
            print('---===== Learning Successful =====---\n')
            break
    numpy.save(norm_encode_path, neural.encode_w)
    numpy.save(norm_decode_path, neural.decode_w)


def test(neural):
    neural.encode_weights = numpy.load(encode_path)
    neural.decode_weights = numpy.load(decode_path)
    pic_num = randint(1, 200)
    pic = mpimg.imread(images_path + f"{pic_num}.png")
    test = neural.test(pic)
    show(pic, test)


def archive(neural):
    neural.encode_weights = numpy.load(encode_path)
    num = randint(1, 200)
    img = mpimg.imread(images_path + f"{num}.png")
    encode_img = neural.encode(img)
    with open(f'{num}.pickle', 'wb') as f:
        pickle.dump(encode_img, f)


def unarchive(neural):
    neural.decode_weights = numpy.load(decode_path)
    name = int(input('Filename input: \n'))
    with open(f'{name}.pickle', 'rb') as f:
        pic_e = pickle.load(f)
    pic_d = neural.decode(pic_e)
    plt.imshow(pic_d)
    plt.show()


def multiplication(n, m):
    rows = len(m)
    result = cp.deepcopy(m)
    for i in range(rows):
        for j in range(len(m[i])):
            result[i][j] = m[i][j] * n
    return result


def subtraction(m1, m2):
    rows_m1, columns_m1 = len(m1), len(m1[0])
    rows_m2, columns_m2 = len(m2), len(m2[0])
    if rows_m1 == rows_m2 and columns_m1 == columns_m2:
        result = cp.deepcopy(m1)
        for i in range(rows_m1):
            for j in range(len(m1[i])):
                result[i][j] = m1[i][j] - m2[i][j]
        return result
    else:
        print('!!! Subtraction Error !!!\n')


def transposition(m):
    rows, columns = len(m), len(m[0])
    result = [[0] * rows for _ in range(columns)]
    for i in range(len(m)):
        for j in range(len(m[i])):
            result[j][i] = m[i][j]
    return result


def find_e(m):
    length = len(m)
    new_matrix = []
    for i in range(length):
        temp = 0
        for j in range(len(m[i])):
            temp += m[i][j] ** 2
        new_matrix.append(temp)
    return sum(new_matrix)


def normalize_weights(w):
    length = len(w)
    for i in range(length):
        for j in range(len(w[i])):
            w[i][j] /= norm(w[:, j], 2)


def mult(matrix1, matrix2):
    return list(
        list(sum([matrix1[i][k] * matrix2[k][j] for k in range(len(matrix2))]) for j in range(len(matrix2[0]))) for i
        in range(len(matrix1)))


def j_find(matrix, j):
    result = []
    print(len(matrix[0]), j)
    for i in range(len(matrix)):
        result.append(matrix[i][j])
    return result


def fullyconnected(tensor, input_size, out_size):
    W = tf.Variable(tf.truncated_normal([input_size, out_size], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([out_size], stddev=0.1))
    return tf.nn.tanh(tf.matmul(tensor, W) + b)


def batchnorm(tensor, size):
    batch_mean, batch_var = tf.nn.moments(tensor, [0])
    beta = tf.Variable(tf.zeros([size]))
    scale = tf.Variable(tf.ones([size]))
    return tf.nn.batch_normalization(tensor, batch_mean, batch_var, beta, scale, 0.001)


def get_one(i, sz):
    res = np.zeros(sz)
    res[i] = 1
    return res
