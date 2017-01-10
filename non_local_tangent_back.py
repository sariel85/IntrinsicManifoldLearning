from __future__ import print_function

import numpy
import theano
import theano.tensor as T
import theano.tensor.nlinalg

class non_local_tangent_net(object):

    def __init__(self, input_base, dim_measurements=2, dim_intrinsic=2, n_hidden_tangent=10, n_hidden_int=10, intrinsic_variance=0.01, measurement_variance=0.01):

        self.dim_intrinsic = dim_intrinsic
        self.dim_measurements = dim_measurements
        self.n_hidden_tangent = n_hidden_tangent
        self.n_hidden_int = n_hidden_int
        self.dim_jacobian = self.dim_intrinsic*self.dim_measurements
        self.n_points = input_base.shape[0]
        self.intrinsic_variance = intrinsic_variance
        self.measurement_variance = measurement_variance
        self.dim_jacobian = self.dim_intrinsic*self.dim_measurements
        self.dim_jacobian_int = self.dim_intrinsic*self.dim_intrinsic

        initial_W_tangent_1 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_tangent + self.dim_measurements)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_tangent + self.dim_measurements)),
                size=(self.n_hidden_tangent, self.dim_measurements)
            ),
            dtype=theano.config.floatX
        )

        initial_W_tangent_2 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_tangent + self.n_hidden_tangent)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_tangent + self.n_hidden_tangent)),
                size=(self.n_hidden_tangent, self.n_hidden_tangent)
            ),
            dtype=theano.config.floatX
        )

        initial_W_tangent_3 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_jacobian + self.n_hidden_tangent)),
                high=4 * numpy.sqrt(6. / (self.dim_jacobian + self.n_hidden_tangent)),
                size=(self.dim_jacobian, self.n_hidden_tangent)
            ),
            dtype=theano.config.floatX
        )

        initial_W_int_1 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_int + self.dim_measurements)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_int + self.dim_measurements)),
                size=(self.n_hidden_int, self.dim_measurements)
            ),
            dtype=theano.config.floatX
        )

        initial_W_int_2 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_int + self.n_hidden_int)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_int + self.n_hidden_int)),
                size=(self.n_hidden_int, self.n_hidden_int)
            ),
            dtype=theano.config.floatX
        )

        initial_W_int_3 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_jacobian_int + self.n_hidden_int)),
                high=4 * numpy.sqrt(6. / (self.dim_jacobian_int + self.n_hidden_int)),
                size=(self.dim_jacobian_int, self.n_hidden_int)
            ),
            dtype=theano.config.floatX
        )


        self.coeffs = theano.shared(value=numpy.zeros((self.dim_intrinsic, self.n_points), dtype=theano.config.floatX), name='coeffs', borrow=False)

        self.W_tangent_1 = theano.shared(value=initial_W_tangent_1, name='W_tangent_1', borrow=False)

        self.W_tangent_2 = theano.shared(value=initial_W_tangent_2, name='W_tangent_2', borrow=False)

        self.W_tangent_3 = theano.shared(value=initial_W_tangent_3, name='W_tangent_3', borrow=False)

        self.W_int_1 = theano.shared(value=initial_W_int_1, name='W_int_1', borrow=False)

        self.W_int_2 = theano.shared(value=initial_W_int_2, name='W_int_2', borrow=False)

        self.W_int_3 = theano.shared(value=initial_W_int_3, name='W_int_3', borrow=False)

        self.b_tangent_1 = theano.shared(value=numpy.zeros((self.n_hidden_tangent, ), dtype=theano.config.floatX), name='b_tangent_1', borrow=False)

        self.b_tangent_2 = theano.shared(value=numpy.zeros((self.n_hidden_tangent, ), dtype=theano.config.floatX), name='b_tangent_2', borrow=False)

        self.b_tangent_3 = theano.shared(value=numpy.zeros((self.dim_jacobian, ), dtype=theano.config.floatX), name='b_tangent_3', borrow=False)

        self.b_int_1 = theano.shared(value=numpy.zeros((self.n_hidden_int,), dtype=theano.config.floatX), name='b_int_1', borrow=False)

        self.b_int_2 = theano.shared(value=numpy.zeros((self.n_hidden_int,), dtype=theano.config.floatX), name='b_int_2', borrow=False)

        self.b_int_3 = theano.shared(value=numpy.zeros((self.dim_jacobian_int,), dtype=theano.config.floatX), name='b_int_3', borrow=False)

    def get_tangent_hidden_1(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_tangent_1, inputs).T + self.b_tangent_1.T).T

    def get_tangent_hidden_2(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_tangent_2, inputs).T + self.b_tangent_2.T).T

    def get_tangent_output(self, inputs):
        return (T.dot(self.W_tangent_3, inputs).T + self.b_tangent_3.T).T

    def get_int_hidden_1(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_int_1, inputs).T + self.b_int_1.T).T

    def get_int_hidden_2(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_int_2, inputs).T + self.b_int_2.T).T

    def get_int_output(self, inputs):
        return (T.dot(self.W_int_3, inputs).T + self.b_int_3.T).T


    def get_jacobian(self, inputs_base):
        tengent_hidden_1 = self.get_tangent_hidden_1(inputs_base)
        tengent_hidden_2 = self.get_tangent_hidden_2(tengent_hidden_1)
        jacobian = T.reshape(self.get_tangent_output(tengent_hidden_2), (self.dim_measurements, self.dim_intrinsic))
        return jacobian

    def get_jacobian_int(self, inputs_base):
        int_hidden_1 = self.get_int_hidden_1(inputs_base)
        int_hidden_2 = self.get_int_hidden_2(int_hidden_1)
        jacobian_int = T.reshape(self.get_int_output(int_hidden_2), (self.dim_intrinsic, self.dim_intrinsic))
        return jacobian_int


    def get_cost(self, inputs_base, inputs_step, coeffs):
        jacobian = self.get_jacobian(inputs_base.T)
        jacobian_int = self.get_jacobian_int(inputs_base.T)
        cov_mat = T.dot(jacobian.T, jacobian)


        cost = ((T.sum((T.dot(jacobian, coeffs.T) - (inputs_step.T-inputs_base.T)) ** 2, 0)/self.measurement_variance+T.log(T.nlinalg.det(T.dot(jacobian.T,jacobian)))-T.log(T.nlinalg.det(T.dot(jacobian_int.T,jacobian_int)))+(T.sum(T.dot(jacobian_int, coeffs.T) ** 2, 0))/self.intrinsic_variance)*self.measurement_variance).mean()
        return cost

    def get_cost_rec(self, inputs_base, inputs_step, coeffs):
        jacobian = self.get_jacobian(inputs_base.T)
        #jacobian_int = self.get_jacobian_int(inputs_base.T)
        cost = (T.sum((T.dot(jacobian, coeffs.T) - (inputs_step.T-inputs_base.T)) ** 2, 0)).mean()
        return cost

    def train_net(self, inputs_base, inputs_step, coeffs):

        return cost







