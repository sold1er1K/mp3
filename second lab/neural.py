import numpy
import tf
import matplotlib.pyplot as plt
import json


class NeuralNetwork:
    def __init__(self, data):
        self.data = data
        self.weights = numpy.zeros((784, 784))
        self.standard = numpy.copy(data)
        self.count = 0

    def learn(self):
        for i in range(10):
            alpha = 1 / ((self.data[i].T @ self.data[i]) - (self.data[i].T @ self.weights @ self.data[i]))
            beta = self.weights @ self.data[i] - self.data[i]
            self.weights += alpha * (beta @ beta.T)
        for (i, j), element in numpy.ndenumerate(self.weights):
            self.weights[i][i] = 0

    def test(self, number):
        data_corrupted = corrupted(number)
        data_output = self.asinc(data_corrupted)
        show(data_output, data_corrupted)

    def test_from_file(self, number):
        with open(f'numbers by json/{number}.json', 'r') as f:
            data_load = json.load(f)
            data = bipolar(numpy.array(data_load).reshape(784, 1))
        data_corrupted = corrupted(data)
        data_output = self.asinc(data_corrupted)
        show(data_output, data_corrupted)

    def get_matrices(self):
        max_sentence_len = numpy.max([len(x) for x in sentences])

        x = numpy.zeros((len(sentences), max_sentence_len, len(chars)), dtype=numpy.bool)
        y = numpy.zeros((len(sentences), max_sentence_len, len(chars)), dtype=numpy.bool)
        for i, sentence in enumerate(sentences):
            char_seq = (START_CHAR + sentence + END_CHAR).ljust(
                max_sentence_len + 1, PADDING_CHAR)
        for t in range(max_sentence_len):
            x[i, t, :] = char_vectors[char_seq[t]]
        y[i, t, :] = char_vectors[char_seq[t + 1]]
        return x, y

    def asinc(self, img):
        i = 0
        sign = numpy.vectorize(lambda x: -1 if x < 0 else 1)
        while True:
            temp_rez = self.weights @ img
            temp_rez = sign(temp_rez)
            if not compare(temp_rez, stand):
                img[i, 0] = temp_rez[i, 0]
                i += 1
                if i == 784:
                    print("Unknown")
                    break
        self.count += 1
        if self.count == len(self.data):
            self.count = 0
        return temp_rez

    def _encoder(self):
        sizes = [self.input_space] + self.middle_layers + [self.latent_space]
        self.encoder_layers = [self.input_x]
        for i in range(len(sizes) - 1):
            with tf.variable_scope('layer-%s' % i):
                linear = linear_layer(self.encoder_layers[-1], sizes[i], sizes[i + 1])
        self.encoder_layers.append(self.activation_fn(linear))

    def show(self, output, corrupted):
        plt.rcParams["figure.figsize"] = (10, 10)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(corrupted.reshape(28, 28), cmap='gray')
        axs[1].imshow(output.reshape(28, 28), cmap='gray')
        axs[0].set_title("Вход")
        axs[1].set_title("Распознано")
        plt.show()

    def _decoder(self, tensor):
        sizes = [self.latent_space] + self.middle_layers[::-1]
        decoder_layers = [tensor]
        for i in range(len(sizes) - 1):
            with tf.variable_scope('layer-%s' % i):
                linear = linear_layer(decoder_layers[-1], sizes[i], sizes[i + 1])
        decoder_layers.append(self.activation_fn(linear))
        with tf.variable_scope('output-layer'):
            linear = linear_layer(decoder_layers[-1], sizes[-1], self.input_space)
        decoder_layers.append(linear)
        return decoder_layers

    def _discriminator(self, tensor, sizes):
        sizes = [self.latent_space] + sizes + [1]
        disc_layers = [tensor]
        for i in range(len(sizes) - 1):
            with tf.variable_scope('layer-%s' % i):
                linear = linear_layer(disc_layers[-1], sizes[i], sizes[i + 1])
        disc_layers.append(self.activation_fn(linear))
        with tf.variable_scope('class-layer'):
            linear = linear_layer(disc_layers[-1], sizes[-1], self.input_space)


def compare(inp, out):
    return out == inp


def corrupted(data):
    c = numpy.copy(data)
    inv = numpy.random.binomial(1, 0.25, len(data))
    for index, vector in enumerate(data):
        if inv[index]:
            c[index] = -1 * vector
    return c


def bipolar(v):
    result = []
    for i in range(len(v)):
        if v[i][0] == 255:
            result.append([1])
        else:
            result.append([-1])
    return numpy.array(result)


def load(num):
    with open(f'numbers by json/{num}.json', 'r') as f:
        data_load = json.load(f)
        return bipolar(numpy.array(data_load).reshape(s, 1))


def batch_gen(data, batch_n):
    inds = range(data.shape[0])
    numpy.random.shuffle(inds)
    for i in range(data.shape[0] / batch_n):
        ii = inds[i*batch_n:(i+1)*batch_n]
        yield data[ii, :]


def he_initializer(size):
    return tf.random_normal_initializer(mean=0.0,
    stddev=numpy.sqrt(1./size), seed=None, dtype=tf.float32)


def linear_layer(tensor, input_size, out_size, init_fn=he_initializer,):
    W = tf.get_variable('W', shape=[input_size, out_size],
    initializer=init_fn(input_size))
    b = tf.get_variable('b', shape=[out_size],
    initializer=tf.constant_initializer(0.1))
    return tf.add(tf.matmul(tensor, W), b)


def sample_prior(loc=0., scale=1., size=(64, 10)):
    return numpy.tanh(numpy.random.normal(loc=loc, scale=scale, size=size))


def discriminator(x):
    d_h1 = tf.nn.tanh(tf.add(
    tf.matmul(x, weights['w1']), weights['b1']))
    d_h2 = tf.nn.tanh(tf.add(
    tf.matmul(d_h1, weights['w2']), weights['b2']))
    logits = tf.add(tf.matmul(d_h2, weights['w3']), weights['b3'])
    return logits

