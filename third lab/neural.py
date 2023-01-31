import numpy
import os


class NeuralNetwork:
    def __init__(self, m):
        self.a = 1e-2
        self.max = 1e-6
        self.it = 1000000
        self.count = 4
        self.matrix = m
        self.first_l = numpy.random.uniform(-1, 1, (self.matrix.shape[1], self.matrix.shape[0]))
        self.second_l = numpy.random.uniform(-1, 1, (self.matrix.shape[0], 1))
        self.jordan = numpy.zeros((1, 1))
        self.error = 0

    def learn(self):
        while self.max > self.error:
            self.error = 0
            numpy.apply_along_axis(self.vector_train, 1, self.matrix)
            for i, element in numpy.ndenumerate(numpy.apply_along_axis(self.calc_error, 1, self.matrix)):
                self.error = self.error + element
        self.test()

    def test(self):
        numpy.apply_along_axis(self.process_vector, 1, self.matrix)
        last = self.matrix[self.matrix.shape[0]-1]
        last[last.size - 1] = 0
        i = 0
        while i < self.count:
            print("Next:", self.process_vector(last))
            last[last.size-1] = self.process_vector(last)
            last = numpy.append(last, 0)
            i = i + 1
        print("Matrix \n", self.matrix)
        print("First layer weights\n", self.first_l)
        print("Second layer weights\n", self.second_l)
        print("Error ", self.error)

    def calc_error(self, vector):
        inputs = numpy.append(vector[:vector.size-1], self.jordan)
        first_net = inputs @ self.first_l
        first_out = activation(first_net)

        second_net = first_out @ self.second_l
        second_out = activation(second_net)

        return 1/2 * ((vector[vector.size-1] - second_out[0]) ** 2)

    def old_train(self):
        x = images
        gloss = 0.69
        for i in range(1000):
            batch_x, _ = train.next_batch(self.batch_size)
        if gloss > numpy.log(self.p):
            gloss, _ = sess.run([self.enc_loss, self.train_generator],
                                feed_dict={self.input_x: batch_x})
        else:
            batch_z = sample_prior(scale=1.0, size=(len(x), self.space))
        gloss, _ = sess.run([self.enc_loss, self.train_discriminator], feed_dict={self.input_x: batch_x, self.z_tensor: batch_z})
        if i % 100 == 0:
            gtd = run(generated, feed_dict={z_tensor: sample_prior(size=(4, 10))})
        plot_mnist(reshape([4, 28, 28]), [1, 4])

    def old_train(self, size=100, training=10, step=5):
        for epoch in range(training):
            avg_cost = 0.
        total_batch = int(n_samples / size)
        for i in range(total_batch):
            xs, _ = mnist.train.next_batch(size)
        _, c = sess.run((optimizer, cost), feed_dict={x: xs})
        avg_cost += c / n_samples * size
        if epoch % display_step == 0:
            print("Epoch: %04d\tcost: %.9f" % (epoch + 1, avg_cost))

    def on_train_begin(self, logs={}):
        self.epoch = 0
        if os.path.isfile(out):
            os.remove(out)

    def sample(self, preds, temperature=1.0):
        preds = numpy.asarray(preds).astype('float64')
        preds = numpy.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / numpy.sum(exp_preds)
        probas = numpy.random.multinomial(1, preds, 1)
        return numpy.argmax(probas)

    def sample_one(self, T):
        result = T
        while len(result) < 500:
            Xsampled = numpy.zeros((1, len(result), num_chars))
        for t, c in enumerate(list(result)):
            Xsampled[0, t, :] = self.char_vectors[c]
        ysampled = self.model.predict(Xsampled, batch_size=1)[0, :]
        yv = ysampled[len(result) - 1, :]
        selected_char = indices_to_chars[self.sample(yv, T)]
        if selected_char == T:
            result = result + selected_char
        return result

    def second(self, target, second_out, second_net, first_out):
        weights = numpy.zeros((self.second_l.shape[0], self.second_l.shape[1]))
        for (i, j), element in numpy.ndenumerate(weights):
            weights[i, j] = - (target - second_out) * function_der(second_net) * first_out[j]
        return weights

    def first(self, target, second_out, second_net, first_net, inputs):
        weights = numpy.zeros((self.first_l.shape[0], self.first_l.shape[1]))
        for (i, j), element in numpy.ndenumerate(weights):
            der_w_i_j = -(target - second_out) * function_der(second_net) * self.second_l[j, 0] \
                        * function_der(first_net[j]) * inputs[i]
            weights[i, j] = der_w_i_j
        return weights

    def load_w(self, row):
        self.first_l = numpy.load(f"weights/{choose(row)}1.npy")
        self.second_l = numpy.load(f"weights/{choose(row)}2.npy")


def function(z, a=1):
    return z if z >= 0 else a * (numpy.exp(z) - 1)


def function_der(z, a=1):
    return 1 if z > 0 else a * numpy.exp(z)


def activation(m):
    for i, element in numpy.ndenumerate(m):
        m[i] = function(element)
    return m


def derivative(m):
    for (i, j), element in numpy.ndenumerate(m):
        m[i, j] = function_der(element)


def batch_gen(data, batch_n):
    inds = range(data.shape[0])
    numpy.random.shuffle(inds)
    for i in range(data.shape[0] / batch_n):
        ii = inds[i*batch_n:(i+1)*batch_n]
        yield data[ii, :]


def he_initializer(size):
    return random_normal_initializer(mean=0.0, stddev=numpy.sqrt(1./size), seed=None, dtype=tf.float32)


def linear_layer(tensor, input_size, out_size, init_fn=he_initializer,):
    W = get_variable('W', shape=[input_size, out_size],
    initializer=init_fn(input_size))
    b = get_variable('b', shape=[out_size],
    initializer=constant_initializer(0.1))
    return add(matmul(tensor, W), b)


def sample_prior(loc=0., scale=1., s=(64, 10)):
    return numpy.tanh(numpy.random.normal(loc=loc, scale=scale, size=s))
