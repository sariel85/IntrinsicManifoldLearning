from __future__ import print_function

import numpy
import theano
import theano.tensor as T
import theano.tensor.nlinalg

class AutoEncoder(object):

    def __init__(self, measurements, n_intrisic_points=2000, dim_input=None, dim_intrinsic=2, n_hidden_rec=10, n_hidden_gen=10, n_hidden_drift=10, n_hidden_inv=10, batch_size=500):
        # self.eye = numpy.eye(n_intrisic_points)
        self.measurements = measurements
        self.n_intrisic_points = n_intrisic_points
        self.dim_intrinsic = dim_intrinsic
        self.dim_messurments = measurements.shape[0]
        self.dim_input = dim_input
        self.n_hidden_rec = n_hidden_rec
        self.n_hidden_gen = n_hidden_gen
        self.n_hidden_inv = n_hidden_inv
        self.n_hidden_drift = n_hidden_drift
        self.batch_size = batch_size
        # TODO: Get initialization from some other place

        initial_W_rec_1 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_input + self.n_hidden_rec)),
                high=4 * numpy.sqrt(6. / (self.dim_input + self.n_hidden_rec)),
                size=(self.n_hidden_rec, self.dim_input)
            ),
            dtype=theano.config.floatX
        )

        W_rec_1 = theano.shared(value=initial_W_rec_1, name='W_rec_1', borrow=False)

        initial_W_rec_2 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_rec + self.n_hidden_rec)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_rec + self.n_hidden_rec)),
                size=(self.n_hidden_rec, self.n_hidden_rec)
            ),
            dtype=theano.config.floatX
        )

        W_rec_2 = theano.shared(value=initial_W_rec_2, name='W_rec_2', borrow=False)

        initial_W_rec_3 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_intrinsic + self.n_hidden_rec)),
                high=4 * numpy.sqrt(6. / (self.dim_intrinsic + self.n_hidden_rec)),
                size=(self.dim_intrinsic, self.n_hidden_rec)
            ),
            dtype=theano.config.floatX
        )

        W_rec_3 = theano.shared(value=initial_W_rec_3, name='W_rec_3', borrow=False)

        b_rec_1 = theano.shared(value=numpy.zeros((self.n_hidden_rec, ), dtype=theano.config.floatX), name='b_rec_1', borrow=False)

        b_rec_2 = theano.shared(value=numpy.zeros((self.n_hidden_rec, ), dtype=theano.config.floatX), name='b_rec_2', borrow=False)

        b_rec_3 = theano.shared(value=numpy.zeros((self.dim_intrinsic, ), dtype=theano.config.floatX), name='b_rec_3', borrow=False)

        initial_W_gen_1 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_intrinsic + self.n_hidden_gen)),
                high=4 * numpy.sqrt(6. / (self.dim_intrinsic + self.n_hidden_gen)),
                size=(self.n_hidden_gen, self.dim_intrinsic)
            ),
            dtype=theano.config.floatX
        )

        W_gen_1 = theano.shared(value=initial_W_gen_1, name='W_gen_1', borrow=False)

        initial_W_gen_2 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_gen + self.n_hidden_gen)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_gen + self.n_hidden_gen)),
                size=(self.n_hidden_gen, self.n_hidden_gen)
            ),
            dtype=theano.config.floatX
        )

        W_gen_2 = theano.shared(value=initial_W_gen_2, name='W_gen_2', borrow=False)

        initial_W_gen_3 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_messurments + self.n_hidden_gen)),
                high=4 * numpy.sqrt(6. / (self.dim_messurments + self.n_hidden_gen)),
                size=(self.dim_messurments, self.n_hidden_gen)
            ),
            dtype=theano.config.floatX
        )

        W_gen_3 = theano.shared(value=initial_W_gen_3, name='W_gen_3', borrow=False)

        b_gen_1 = theano.shared(value=numpy.zeros((self.n_hidden_gen, ), dtype=theano.config.floatX), name='b_gen_1', borrow=False)

        b_gen_2 = theano.shared(value=numpy.zeros((self.n_hidden_gen, ), dtype=theano.config.floatX), name='b_gen_2', borrow=False)

        b_gen_3 = theano.shared(value=numpy.zeros((self.dim_messurments, ), dtype=theano.config.floatX), name='b_gen_3', borrow=False)

        initial_W_drift_1 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_input + self.n_hidden_drift)),
                high=4 * numpy.sqrt(6. / (self.dim_input + self.n_hidden_drift)),
                size=(self.n_hidden_drift, self.dim_input)
            ),
            dtype=theano.config.floatX
        )

        W_drift_1 = theano.shared(value=initial_W_drift_1, name='W_drift_1', borrow=False)

        initial_W_drift_2 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_drift + self.n_hidden_drift)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_drift + self.n_hidden_drift)),
                size=(self.n_hidden_drift, self.n_hidden_drift)
            ),
            dtype=theano.config.floatX
        )

        W_drift_2 = theano.shared(value=initial_W_drift_2, name='W_drift_2', borrow=False)

        initial_W_drift_3 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_input + self.n_hidden_drift)),
                high=4 * numpy.sqrt(6. / (self.dim_input + self.n_hidden_drift)),
                size=(self.dim_input, self.n_hidden_drift)
            ),
            dtype=theano.config.floatX
        )

        W_drift_3 = theano.shared(value=initial_W_drift_3, name='W_drift_3', borrow=False)

        b_drift_1 = theano.shared(value=numpy.zeros((self.n_hidden_drift, ), dtype=theano.config.floatX), name='b_drift_1', borrow=False)

        b_drift_2 = theano.shared(value=numpy.zeros((self.n_hidden_drift, ), dtype=theano.config.floatX), name='b_drift_2', borrow=False)

        b_drift_3 = theano.shared(value=numpy.zeros((self.dim_input, ), dtype=theano.config.floatX), name='b_drift_3', borrow=False)

        initial_W_inv_1 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_intrinsic + self.n_hidden_inv)),
                high=4 * numpy.sqrt(6. / (self.dim_intrinsic + self.n_hidden_inv)),
                size=(self.n_hidden_inv, self.dim_intrinsic)
            ),
            dtype=theano.config.floatX
        )

        W_inv_1 = theano.shared(value=initial_W_inv_1, name='W_inv_1', borrow=False)

        initial_W_inv_2 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_inv + self.n_hidden_inv)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_inv + self.n_hidden_inv)),
                size=(self.n_hidden_inv, self.n_hidden_inv)
            ),
            dtype=theano.config.floatX
        )

        W_inv_2 = theano.shared(value=initial_W_inv_2, name='W_inv_2', borrow=False)

        initial_W_inv_3 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_intrinsic + self.n_hidden_inv)),
                high=4 * numpy.sqrt(6. / (self.dim_intrinsic + self.n_hidden_inv)),
                size=(self.dim_intrinsic, self.n_hidden_inv)
            ),
            dtype=theano.config.floatX
        )

        W_inv_3 = theano.shared(value=initial_W_inv_3, name='W_inv_3', borrow=False)


        b_inv_1 = theano.shared(value=numpy.zeros((self.n_hidden_inv, ), dtype=theano.config.floatX), name='b_inv_1', borrow=False)

        b_inv_2 = theano.shared(value=numpy.zeros((self.n_hidden_inv, ), dtype=theano.config.floatX), name='b_inv_2', borrow=False)

        b_inv_3 = theano.shared(value=numpy.zeros((self.dim_intrinsic, ), dtype=theano.config.floatX), name='b_inv_3', borrow=False)

        self.W_rec_1 = W_rec_1
        self.b_rec_1 = b_rec_1
        self.W_rec_2 = W_rec_2
        self.b_rec_2 = b_rec_2
        self.W_rec_3 = W_rec_3
        self.b_rec_3 = b_rec_3

        self.W_gen_1 = W_gen_1
        self.b_gen_1 = b_gen_1
        self.W_gen_2 = W_gen_2
        self.b_gen_2 = b_gen_2
        self.W_gen_3 = W_gen_3
        self.b_gen_3 = b_gen_3

        self.W_drift_1 = W_drift_1
        self.b_drift_1 = b_drift_1
        self.W_drift_2 = W_drift_2
        self.b_drift_2 = b_drift_2
        self.W_drift_3 = W_drift_3
        self.b_drift_3 = b_drift_3

        self.W_inv_1 = W_inv_1
        self.b_inv_1 = b_inv_1
        self.W_inv_2 = W_inv_2
        self.b_inv_2 = b_inv_2
        self.W_inv_3 = W_inv_3
        self.b_inv_3 = b_inv_3

    #def get_intrinsic_values_exact(self, eye_matrix):
    #    return T.dot(self.W_intrinsic, eye_matrix)

    def get_inv_hidden_1(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_inv_1, inputs).T + self.b_inv_1.T).T

    def get_inv_hidden_2(self, inv_hidden_1):
        return T.nnet.sigmoid(T.dot(self.W_inv_2, inv_hidden_1).T + self.b_inv_2.T).T

    def get_inv_output(self, inv_hidden_2):
        return (T.dot(self.W_inv_3, inv_hidden_2).T + self.b_inv_3.T).T
        #return T.dot(self.W_inv_3, inv_hidden_2)

    def get_rec_hidden_1(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_rec_1, inputs).T + self.b_rec_1.T).T

    def get_rec_hidden_2(self, rec_hidden_1):
        return T.nnet.sigmoid(T.dot(self.W_rec_2, rec_hidden_1).T + self.b_rec_2.T).T

    def get_rec_output(self, rec_hidden_2):
        return (T.dot(self.W_rec_3, rec_hidden_2).T + self.b_rec_3.T).T

    def get_gen_hidden_1(self, gen_input):
        return T.nnet.sigmoid(T.dot(self.W_gen_1, gen_input).T + self.b_gen_1.T).T

    def get_gen_hidden_2(self, gen_hidden_1):
        return T.nnet.sigmoid(T.dot(self.W_gen_2, gen_hidden_1).T + self.b_gen_2.T).T

    def get_gen_output(self, gen_hidden_2):
        return (T.dot(self.W_gen_3, gen_hidden_2).T + self.b_gen_3.T).T

    def get_drift_hidden_1(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_drift_1, inputs).T + self.b_drift_1.T).T

    def get_drift_hidden_2(self, drift_hidden_1):
        return T.nnet.sigmoid(T.dot(self.W_drift_2, drift_hidden_1).T + self.b_drift_2.T).T

    def get_drift_output(self, drift_hidden_2):
        return (T.dot(self.W_drift_3, drift_hidden_2).T + self.b_drift_3.T).T

    def get_jacobian_gen(self, hidden_1, hidden_2):
        stage_one_1 = T.reshape(self.W_gen_1, (1, self.W_gen_1.shape[0], self.W_gen_1.shape[1]))
        temp_1 = hidden_1 * (1 - hidden_1)
        stage_one_2 = T.reshape(temp_1.T, (temp_1.shape[1], temp_1.shape[0], 1))
        stage_one = stage_one_2 * stage_one_1
        stage_two_1 = T.reshape(self.W_gen_2, (1, self.W_gen_2.shape[0], self.W_gen_2.shape[1]))
        temp_2 = hidden_2 * (1 - hidden_2)
        stage_two_2 = T.reshape(temp_2.T, (temp_2.shape[1], temp_2.shape[0], 1))
        stage_two = stage_two_2 * stage_two_1
        stage_three = T.reshape(self.W_gen_3, (1, self.W_gen_3.shape[0], self.W_gen_3.shape[1]))
        return T.batched_dot(T.extra_ops.repeat(stage_three, hidden_1.shape[1], axis=0), T.batched_dot(stage_two, stage_one))

    def get_jacobian_inv(self, hidden_1, hidden_2):
        stage_one_1 = T.reshape(self.W_inv_1, (1, self.W_inv_1.shape[0], self.W_inv_1.shape[1]))
        temp_1 = hidden_1 * (1 - hidden_1)
        stage_one_2 = T.reshape(temp_1.T, (temp_1.shape[1], temp_1.shape[0], 1))
        stage_one = stage_one_2 * stage_one_1
        stage_two_1 = T.reshape(self.W_inv_2, (1, self.W_inv_2.shape[0], self.W_inv_2.shape[1]))
        temp_2 = hidden_2 * (1 - hidden_2)
        stage_two_2 = T.reshape(temp_2.T, (temp_2.shape[1], temp_2.shape[0], 1))
        stage_two = stage_two_2 * stage_two_1
        stage_three = T.reshape(self.W_inv_3, (1, self.W_inv_3.shape[0], self.W_inv_3.shape[1]))
        return T.batched_dot(T.extra_ops.repeat(stage_three, hidden_1.shape[1], axis=0), T.batched_dot(stage_two, stage_one))

    def get_cost(self, dim_intrinsic, intrinsic_variance, measurement_variance, inputs_base, inputs_step, measurements, initial_guess_base):

        rec_hidden_1_base = self.get_rec_hidden_1(inputs_base.T)
        rec_hidden_2_base = self.get_rec_hidden_2(rec_hidden_1_base)
        rec_output_base = self.get_rec_output(rec_hidden_2_base)

        rec_hidden_1_step = self.get_rec_hidden_1(inputs_step.T)
        rec_hidden_2_step = self.get_rec_hidden_2(rec_hidden_1_step)
        rec_output_step = self.get_rec_output(rec_hidden_2_step)

        gen_hidden_1_base = self.get_gen_hidden_1(rec_output_base)
        gen_hidden_2_base = self.get_gen_hidden_2(gen_hidden_1_base)
        gen_output_base = self.get_gen_output(gen_hidden_2_base)

        drift_hidden_1 = self.get_drift_hidden_1(inputs_base.T)
        drift_hidden_2 = self.get_drift_hidden_2(drift_hidden_1)
        drift_output = self.get_drift_output(drift_hidden_2)

        drift_hidden_1_rec = self.get_rec_hidden_1(drift_output)
        drift_hidden_2_rec = self.get_rec_hidden_2(drift_hidden_1_rec)
        drift_output_rec = self.get_rec_output(drift_hidden_2_rec)
        drift_hidden_1_inv = self.get_inv_hidden_1(drift_output_rec)
        drift_hidden_2_inv = self.get_inv_hidden_2(drift_hidden_1_inv)
        drift_output_inv = self.get_inv_output(drift_hidden_2_inv)

        inv_hidden_1_base = self.get_inv_hidden_1(rec_output_base)
        inv_hidden_2_base = self.get_inv_hidden_2(inv_hidden_1_base)
        inv_output_base = self.get_inv_output(inv_hidden_2_base)

        inv_hidden_1_step = self.get_inv_hidden_1(rec_output_step)
        inv_hidden_2_step = self.get_inv_hidden_2(inv_hidden_1_step)
        inv_output_step = self.get_inv_output(inv_hidden_2_step)

        gen_func_jacobian = self.get_jacobian_gen(gen_hidden_1_base, gen_hidden_2_base)
        inv_func_jacobian = self.get_jacobian_inv(inv_hidden_1_base, inv_hidden_2_base)

        gen_func_jacobian_squared = T.batched_dot(gen_func_jacobian.dimshuffle((0, 2, 1)), gen_func_jacobian)
        inv_func_jacobian_squared = T.batched_dot(inv_func_jacobian, inv_func_jacobian.dimshuffle((0, 2, 1)))

        if dim_intrinsic == 2:
            det_line_1 = gen_func_jacobian_squared[:, 0, 0] * gen_func_jacobian_squared[:, 1, 1] - gen_func_jacobian_squared[:, 0, 1] * gen_func_jacobian_squared[:, 1, 0]
            det_line_1_inv = inv_func_jacobian_squared[:, 0, 0] * inv_func_jacobian_squared[:, 1, 1] - inv_func_jacobian_squared[:, 0, 1] * inv_func_jacobian_squared[:, 1, 0]
        if dim_intrinsic == 3:
            det_line_1 = gen_func_jacobian_squared[:, 0, 0]*(gen_func_jacobian_squared[:, 1, 1] * gen_func_jacobian_squared[:, 2, 2] - gen_func_jacobian_squared[:, 1, 2] * gen_func_jacobian_squared[:, 2, 1])\
                         - gen_func_jacobian_squared[:, 0, 1]*(gen_func_jacobian_squared[:, 1, 0] * gen_func_jacobian_squared[:, 2, 2] - gen_func_jacobian_squared[:, 1, 2] * gen_func_jacobian_squared[:, 2, 0])\
                         + gen_func_jacobian_squared[:, 0, 2]*(gen_func_jacobian_squared[:, 1, 0] * gen_func_jacobian_squared[:, 2, 1] - gen_func_jacobian_squared[:, 1, 1] * gen_func_jacobian_squared[:, 2, 0])
            det_line_1_inv = inv_func_jacobian_squared[:, 0, 0]*(inv_func_jacobian_squared[:, 1, 1] * inv_func_jacobian_squared[:, 2, 2] - inv_func_jacobian_squared[:, 1, 2] * inv_func_jacobian_squared[:, 2, 1])\
                            - inv_func_jacobian_squared[:, 0, 1]*(inv_func_jacobian_squared[:, 1, 0] * inv_func_jacobian_squared[:, 2, 2] - inv_func_jacobian_squared[:, 1, 2] * inv_func_jacobian_squared[:, 2, 0])\
                            + inv_func_jacobian_squared[:, 0, 2]*(inv_func_jacobian_squared[:, 1, 0] * inv_func_jacobian_squared[:, 2, 1] - inv_func_jacobian_squared[:, 1, 1] * inv_func_jacobian_squared[:, 2, 0])

        det_line_2 = (1/2)*T.log(det_line_1)
        det_line_2_inv = (1/2)*T.log(det_line_1_inv)

        gen_hidden_1_base_init = self.get_gen_hidden_1(initial_guess_base.T)
        gen_hidden_2_base_init = self.get_gen_hidden_2(gen_hidden_1_base_init)
        gen_output_base_init = self.get_gen_output(gen_hidden_2_base_init)

        inv_hidden_1_base_init = self.get_inv_hidden_1(initial_guess_base.T)
        inv_hidden_2_base_init = self.get_inv_hidden_2(inv_hidden_1_base_init)
        inv_output_base_init = self.get_inv_output(inv_hidden_2_base_init)


        # Cost for pretraining Starting Point
        cost_rec_pretrain = (1/2)*T.sum((rec_output_base - initial_guess_base.T)**2, 0)

        # Cost of Pretraining Reconstruction
        cost_gen_pretrain = (1/2)*T.sum((gen_output_base_init - measurements.T)**2, 0)

        # Cos of Pretaining Inversion
        cost_inv_pretrain = (1/2)*T.sum((inv_output_base_init - initial_guess_base.T)**2, 0)

        # Cost Measurement Reconstruction vs Noise
        cost_measurements_reg = 1/2*T.sum((gen_output_base - measurements.T)**2, 0)/measurement_variance

        # Cost of Measurement Function Determinant
        cost_measurements_det = -det_line_2_inv + det_line_2

        # Cost of measurement assuming x given
        cost_gen = cost_measurements_reg + det_line_2 + T.sum((rec_output_step - rec_output_base)**2, 0)/intrinsic_variance

        # Cost of Diffusion in addition to drift
        cost_intrinsic = (1/2)*T.sum((inv_output_step - drift_output_inv)**2, 0)/intrinsic_variance

        # Cost of measurement assuming x given
        cost_inv = cost_intrinsic - det_line_2_inv

        # Cost of measurement assuming x given
        cost_total = cost_intrinsic + cost_measurements_reg + cost_measurements_det

        # Likelihood of Intrinsic Process
        cost_drift = (1/2)*T.sum((inv_output_step - drift_output_inv)**2, 0)/intrinsic_variance

        cost_drift_pretrain = (1/2)*T.sum((drift_output - inputs_step.T)**2, 0)/intrinsic_variance

        return rec_output_base, gen_output_base_init, gen_output_base, inv_output_base_init, inv_output_base, drift_output_inv, cost_rec_pretrain, cost_gen_pretrain, cost_gen, cost_inv_pretrain, cost_drift_pretrain, cost_drift, cost_inv, cost_intrinsic, cost_measurements_reg, cost_total