from __future__ import print_function

import numpy
import theano
import theano.tensor as T
import theano.tensor.nlinalg
import matplotlib.pyplot as plt

class non_local_tangent_net(object):

    def __init__(self, dim_measurements=2, dim_intrinsic=2, n_hidden_tangent=10, n_hidden_int=10, n_hidden_drift=10, intrinsic_variance=0.01, measurement_variance=0.01):

        self.input_base_Theano = T.dmatrix('input_base_Theano')
        self.input_step_Theano = T.dmatrix('input_step_Theano')
        self.input_coeff_Theano = T.dmatrix('input_coeff_Theano')
        self.dim_intrinsic = dim_intrinsic
        self.dim_measurements = dim_measurements
        self.n_hidden_tangent = n_hidden_tangent
        self.n_hidden_int = n_hidden_int
        self.n_hidden_drift = n_hidden_drift

        self.dim_jacobian = self.dim_intrinsic*self.dim_measurements
        #self.n_points = input_base.shape[0]
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

        initial_W_drift_1 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_drift + self.dim_measurements)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_drift + self.dim_measurements)),
                size=(self.n_hidden_drift, self.dim_measurements)
            ),
            dtype=theano.config.floatX
        )

        initial_W_drift_2 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_hidden_drift + self.n_hidden_drift)),
                high=4 * numpy.sqrt(6. / (self.n_hidden_drift + self.n_hidden_drift)),
                size=(self.n_hidden_drift, self.n_hidden_drift)
            ),
            dtype=theano.config.floatX
        )

        initial_W_drift_3 = numpy.asarray(
            numpy.random.uniform(
                low=-4 * numpy.sqrt(6. / (self.dim_measurements + self.n_hidden_drift)),
                high=4 * numpy.sqrt(6. / (self.dim_measurements + self.n_hidden_drift)),
                size=(self.dim_measurements, self.n_hidden_drift)
            ),
            dtype=theano.config.floatX
        )


        self.W_tangent_1 = theano.shared(value=initial_W_tangent_1, name='W_tangent_1', borrow=True)

        self.W_tangent_2 = theano.shared(value=initial_W_tangent_2, name='W_tangent_2', borrow=True)

        self.W_tangent_3 = theano.shared(value=initial_W_tangent_3, name='W_tangent_3', borrow=True)

        self.W_int_1 = theano.shared(value=initial_W_int_1, name='W_int_1', borrow=True)

        self.W_int_2 = theano.shared(value=initial_W_int_2, name='W_int_2', borrow=True)

        self.W_int_3 = theano.shared(value=initial_W_int_3, name='W_int_3', borrow=True)

        self.b_tangent_1 = theano.shared(value=numpy.zeros((self.n_hidden_tangent, ), dtype=theano.config.floatX), name='b_tangent_1', borrow=True)

        self.b_tangent_2 = theano.shared(value=numpy.zeros((self.n_hidden_tangent, ), dtype=theano.config.floatX), name='b_tangent_2', borrow=True)

        self.b_tangent_3 = theano.shared(value=numpy.zeros((self.dim_jacobian, ), dtype=theano.config.floatX), name='b_tangent_3', borrow=True)

        self.b_int_1 = theano.shared(value=numpy.zeros((self.n_hidden_int,), dtype=theano.config.floatX), name='b_int_1', borrow=True)

        self.b_int_2 = theano.shared(value=numpy.zeros((self.n_hidden_int,), dtype=theano.config.floatX), name='b_int_2', borrow=True)

        #self.b_int_3 = theano.shared(value=numpy.zeros((self.dim_jacobian_int,), dtype=theano.config.floatX), name='b_int_3', borrow=True)

        self.b_int_3 = theano.shared(value=numpy.array([1., 0, 0, 1.], dtype=theano.config.floatX), name='b_int_3', borrow=True)


        self.W_drift_1 = theano.shared(value=initial_W_drift_1, name='W_drift_1', borrow=True)

        self.W_drift_2 = theano.shared(value=initial_W_drift_2, name='W_drift_2', borrow=True)

        self.W_drift_3 = theano.shared(value=initial_W_drift_3, name='W_drift_3', borrow=True)

        self.b_drift_1 = theano.shared(value=numpy.zeros((self.n_hidden_drift,), dtype=theano.config.floatX), name='b_drift_1', borrow=True)

        self.b_drift_2 = theano.shared(value=numpy.zeros((self.n_hidden_drift,), dtype=theano.config.floatX), name='b_drift_2', borrow=True)

        self.b_drift_3 = theano.shared(value=numpy.zeros((self.dim_measurements,), dtype=theano.config.floatX), name='b_drift_3', borrow=True)



        self.get_jacobian_val = theano.function(
            inputs=[self.input_base_Theano],
            outputs=self.get_jacobian(self.input_base_Theano),
        )

        self.get_jacobian_int_val = theano.function(
            inputs=[self.input_base_Theano],
            outputs=self.get_jacobian_int(self.input_base_Theano),
        )

        self.get_drift_val = theano.function(
            inputs=[self.input_base_Theano],
            outputs=self.get_drift(self.input_base_Theano),
        )



    def get_drift_hidden_1(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_drift_1, inputs).T + self.b_drift_1.T).T

    def get_drift_hidden_2(self, inputs):
        return T.nnet.sigmoid(T.dot(self.W_drift_2, inputs).T + self.b_drift_2.T).T

    def get_drift_output(self, inputs):
        return (T.dot(self.W_drift_3, inputs).T + self.b_drift_3.T).T

    def get_drift(self, inputs_base):
        drift_hidden_1 = self.get_drift_hidden_1(inputs_base)
        drift_hidden_2 = self.get_drift_hidden_2(drift_hidden_1)
        return self.get_drift_output(drift_hidden_2).T


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
        jacobian = T.reshape(self.get_tangent_output(tengent_hidden_2).T, (inputs_base.shape[1], self.dim_measurements, self.dim_intrinsic))
        return jacobian

    def get_jacobian_int(self, inputs_base):
        int_hidden_1 = self.get_int_hidden_1(inputs_base)
        int_hidden_2 = self.get_int_hidden_2(int_hidden_1)
        jacobian_int = T.reshape(self.get_int_output(int_hidden_2).T, (inputs_base.shape[1],self.dim_intrinsic, self.dim_intrinsic))
        return jacobian_int

    def get_cost(self, inputs_base, inputs_step, coeffs):
        drift = self.get_drift(inputs_base.T)
        jacobian = self.get_jacobian(inputs_base.T + drift.T)
        jacobian_int = self.get_jacobian_int(inputs_base.T + drift.T)

        jacobian_squared = T.batched_dot(jacobian, jacobian.dimshuffle((0, 2, 1)))
        jacobian_int_squared = T.batched_dot(jacobian_int, jacobian_int.dimshuffle((0, 2, 1)))

        det_jacobian_squared = T.abs_(jacobian_squared[:,0, 0] * jacobian_squared[:,1, 1] - jacobian_squared[:,0, 1] * jacobian_squared[:,1, 0])
        det_jacobian_int_squared = T.abs_(jacobian_int_squared[:,0, 0] * jacobian_int_squared[:,1, 1] - jacobian_int_squared[:, 0, 1] * jacobian_int_squared[:, 1, 0])

        cost_drift = (T.sum((inputs_step - inputs_base-drift) ** 2, 1)).mean()+1*(T.sum(self.W_drift_1** 2)+T.sum(self.W_drift_2** 2)+T.sum(self.W_drift_3** 2))

        cost = (T.sum((T.batched_dot(jacobian, coeffs) - (inputs_step - inputs_base-drift)) ** 2, 1)/self.measurement_variance+T.log(det_jacobian_squared)+T.sum(coeffs ** 2, 1)/self.intrinsic_variance).mean()
        #cost = (T.sum((T.batched_dot(jacobian, coeffs) - (inputs_step - inputs_base-drift)) ** 2, 1)).mean()

        #cost = (T.sum((T.batched_dot(jacobian, coeffs) - (inputs_step-inputs_base)) ** 2, 1)/self.measurement_variance).mean()
        cost_int = (T.log(det_jacobian_squared)-T.log(det_jacobian_int_squared)+T.sum(T.batched_dot(jacobian_int, coeffs) ** 2, 1)/self.intrinsic_variance).mean()

        return cost, cost_int, cost_drift


    def gradient_updates_momentum(self, cost, params, learning_rate, momentum):

        b1 = momentum
        b2 = 0.01*momentum
        e = 1e-8
        updates = []
        grads = T.grad(cost, params)

        i = theano.shared(0.)
        i_t = i + 1.

        fix1 = theano.shared(1.)
        fix2 = theano.shared(1.)

        fix1_T = fix1*(1-b1)
        fix2_T = fix2*(1-b2)

        fix1_fact = 1. - fix1_T
        fix2_fact = 1. - fix2_T
        lr_t = learning_rate * (T.sqrt(fix2_fact) / fix1_fact)
        #lr_t = learning_rate * (1/ fix1_fact)
        for p, g in zip(params, grads):
            g = T.clip(g, -5, 5)
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            #g_t = m_t
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))

        updates.append((fix1, fix1_T))
        updates.append((fix2, fix2_T))
        updates.append((i, i_t))

        return updates

    def gradient_updates_momentum_int(self, cost, params, learning_rate, momentum):

        b1 = momentum
        b2 = 0.01*momentum
        e = 1e-8
        updates = []
        grads = T.grad(cost, params)

        i = theano.shared(0.)
        i_t = i + 1.

        fix1 = theano.shared(1.)
        fix2 = theano.shared(1.)

        fix1_T = fix1*(1-b1)
        fix2_T = fix2*(1-b2)

        fix1_fact = 1. - fix1_T
        fix2_fact = 1. - fix2_T
        #lr_t = learning_rate * (T.sqrt(fix2_fact) / fix1_fact)
        #lr_t = learning_rate * (1/ fix1_fact)
        lr_t = learning_rate
        for p, g in zip(params, grads):
            g = T.clip(g, -5, 5)
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            #g_t = m_t / (T.sqrt(v_t) + e)
            g_t = m_t
            p_t = p - (lr_t * g_t)
            #p_t = p - (learning_rate * g)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))

        updates.append((fix1, fix1_T))
        updates.append((fix2, fix2_T))
        updates.append((i, i_t))

        return updates

    def train_net(self, noisy_sensor_base, noisy_sensor_step):
        max_epoch_drift = 20000
        max_epoch_tangent = 4000
        max_epoch_int = 4000

        n_points = noisy_sensor_base.shape[1]

        n_valid_points = int(numpy.ceil(n_points*0.001))

        n_points = n_points - n_valid_points

        noisy_sensor_valid_base = noisy_sensor_base[:, n_points:]

        noisy_sensor_valid_step = noisy_sensor_step[:, n_points:]

        noisy_sensor_base = noisy_sensor_base[:, :n_points]

        noisy_sensor_step = noisy_sensor_step[:, :n_points]

        coeffs = numpy.random.uniform(low=-1, high=1, size=(self.dim_intrinsic, n_points))

        coeffs_valid = numpy.random.uniform(low=-1, high=1, size=(self.dim_intrinsic, n_valid_points))

        params = [self.W_tangent_1, self.W_tangent_2,
                  self.W_tangent_3, self.b_tangent_1,
                  self.b_tangent_2, self.b_tangent_3]

        params2 = [self.W_int_1, self.W_int_2,
                   self.W_int_3, self.b_int_1,
                   self.b_int_2, self.b_int_3]

        params3 = [self.W_drift_1, self.W_drift_2,
                   self.W_drift_3, self.b_drift_1,
                   self.b_drift_2, self.b_drift_3]

        learning_rate = theano.shared(1e-2)
        momentum = theano.shared(1)
        cost, cost_int, cost_drift = self.get_cost(self.input_base_Theano, self.input_step_Theano, self.input_coeff_Theano)
        updates = self.gradient_updates_momentum(cost_drift, params3, learning_rate, momentum)

        train = theano.function(inputs=[self.input_base_Theano, self.input_step_Theano], outputs=cost_drift, updates=updates)
        train_valid = theano.function(inputs=[self.input_base_Theano, self.input_step_Theano], outputs=cost_drift, updates=None)

        cost_term = []
        cost_term_valid = []
        iteration = 0
        n_batch = 20000
        n_batch = min(n_batch, n_points)

        max_iteration_drift = (n_points/n_batch)*max_epoch_drift

        while iteration < max_iteration_drift:
            point_in_batch = numpy.random.choice(n_points, size=(n_batch), replace=False)
            current_cost = train(noisy_sensor_base[:, point_in_batch].reshape((self.dim_measurements, n_batch)).T, noisy_sensor_step[:, point_in_batch].reshape((self.dim_measurements, n_batch)).T)
            current_valid_cost = train_valid(noisy_sensor_valid_base[:, :].reshape((self.dim_measurements, n_valid_points)).T, noisy_sensor_valid_step[:, :].reshape((self.dim_measurements, n_valid_points)).T,)

            iteration += 1

            print("iteration=", iteration, "learning_rate=", learning_rate.get_value(), "cost:", current_cost.mean(), "cost valid:", current_valid_cost.mean())

            cost_term.append(current_cost.mean())
            cost_term_valid.append(current_valid_cost.mean())

            if iteration % 100 == 0:
                learning_rate.set_value(0.99 * learning_rate.get_value())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cost_term[1:], c='b')
        ax.plot(cost_term_valid[1:], c='r')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cost')
        ax.set_title("Cost vs Epochs - Stage I")
        plt.show(block=False)

        learning_rate = theano.shared(1e-2)
        momentum = theano.shared(1.)
        #cost, cost_int, cost_drift= self.get_cost(self.input_base_Theano, self.input_step_Theano, self.input_coeff_Theano)
        updates = self.gradient_updates_momentum(cost, params, learning_rate, momentum)

        train = theano.function(inputs=[self.input_base_Theano, self.input_step_Theano, self.input_coeff_Theano], outputs=cost, updates=updates)
        train_valid = theano.function(inputs=[self.input_base_Theano, self.input_step_Theano, self.input_coeff_Theano], outputs=cost, updates=None)

        cost_term = []
        cost_term_valid = []
        iteration = 0
        n_batch = 2000
        n_batch = min(n_batch, n_points)

        max_iteration_tangent = (n_points/n_batch)*max_epoch_tangent

        while iteration < max_iteration_tangent:
            if iteration % 100 == 0:
                for i_point in numpy.random.choice(n_points, size=(n_points), replace=False):
                    drift = self.get_drift_val(noisy_sensor_base[:, i_point].reshape((self.dim_measurements, 1)))[0,:]
                    jacobian = self.get_jacobian_val(noisy_sensor_base[:, i_point].reshape((self.dim_measurements, 1))+drift.T)[0,:,:]
                    jacobian_int = self.get_jacobian_int_val(noisy_sensor_base[:, i_point].reshape((self.dim_measurements, 1)))[0,:,:]
                    #coeffs[:, i_point] = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(jacobian.T, jacobian) / self.measurement_variance), jacobian.T), noisy_sensor_step[:, i_point] - noisy_sensor_base[:, i_point]-drift.T) / self.measurement_variance
                    #coeffs[:, i_point] = numpy.sqrt(self.intrinsic_variance)*coeffs[:, i_point]/numpy.linalg.norm(coeffs[:, i_point])
                    coeffs[:, i_point] = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(jacobian.T, jacobian)/self.measurement_variance+numpy.dot(jacobian_int.T, jacobian_int)/self.intrinsic_variance), jacobian.T), noisy_sensor_step[:, i_point]-noisy_sensor_base[:, i_point])/self.measurement_variance

            point_in_batch = numpy.random.choice(n_points, size=(n_batch), replace=False)
            current_cost = train(noisy_sensor_base[:, point_in_batch].reshape((self.dim_measurements, n_batch)).T, noisy_sensor_step[:, point_in_batch].reshape((self.dim_measurements, n_batch)).T, coeffs[:, point_in_batch].reshape((self.dim_intrinsic, n_batch)).T)

            if iteration%100==0:
                for i_point in numpy.random.choice(n_valid_points, size=(n_valid_points), replace=False):
                    drift = self.get_drift_val(noisy_sensor_valid_base[:, i_point].reshape((self.dim_measurements, 1)))[0,:]
                    jacobian = self.get_jacobian_val(noisy_sensor_valid_base[:, i_point].reshape((self.dim_measurements, 1))+drift.T)[0,:,:]
                    jacobian_int = self.get_jacobian_int_val(noisy_sensor_valid_base[:, i_point].reshape((self.dim_measurements, 1)))[0,:,:]
                    #coeffs_valid[:, i_point] = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(jacobian.T, jacobian) / self.measurement_variance), jacobian.T), noisy_sensor_valid_step[:, i_point] - noisy_sensor_valid_base[:, i_point]-drift.T) / self.measurement_variance
                    #coeffs_valid[:, i_point] = numpy.sqrt(self.intrinsic_variance)*coeffs_valid[:, i_point]/numpy.linalg.norm(coeffs_valid[:, i_point])
                    coeffs_valid[:, i_point] = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(jacobian.T, jacobian)/self.measurement_variance+numpy.dot(jacobian_int.T, jacobian_int)/self.intrinsic_variance), jacobian.T), noisy_sensor_valid_step[:, i_point]-noisy_sensor_valid_base[:, i_point])/self.measurement_variance
            current_valid_cost = train_valid(noisy_sensor_valid_base[:, :].reshape((self.dim_measurements, n_valid_points)).T, noisy_sensor_valid_step[:, :].reshape((self.dim_measurements, n_valid_points)).T,coeffs_valid[:, :].reshape((self.dim_intrinsic, n_valid_points)).T)

            iteration += 1

            print("iteration=", iteration, "learning_rate=", learning_rate.get_value(), "cost:", current_cost.mean(), "cost valid:", current_valid_cost.mean())

            cost_term.append(current_cost.mean())
            cost_term_valid.append(current_valid_cost.mean())

            if iteration % 200 == 0:
                learning_rate.set_value(0.95 * learning_rate.get_value())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cost_term[1:], c='b')
        ax.plot(cost_term_valid[1:], c='r')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cost')
        ax.set_title("Cost vs Epochs - Stage I")


        learning_rate = theano.shared(0.01)
        momentum = theano.shared(0.7)
        #cost, cost_int = self.get_cost(self.input_base_Theano, self.input_step_Theano, self.input_coeff_Theano)
        updates = self.gradient_updates_momentum_int(cost_int, params2, learning_rate, momentum)

        train = theano.function(inputs=[self.input_base_Theano, self.input_coeff_Theano], outputs=cost_int, updates=updates)
        train_valid = theano.function(inputs=[self.input_base_Theano, self.input_coeff_Theano], outputs=cost_int, updates=None)

        cost_term_2 = []
        cost_term_valid_2 = []

        iteration = 0
        n_batch = 20000
        n_batch = min(n_batch, n_points)

        max_iteration_int = (n_points/n_batch)*max_epoch_int

        while iteration < max_iteration_int:
            point_in_batch = numpy.random.choice(n_points, size=n_batch, replace=False)
            if iteration % 100 == 0:
                for i_point in point_in_batch:
                    jacobian = self.get_jacobian_val(noisy_sensor_base[:, i_point].reshape((self.dim_measurements, 1)))[0,:,:]
                    jacobian_int = self.get_jacobian_int_val(noisy_sensor_base[:, i_point].reshape((self.dim_measurements, 1)))[0,:,:]
                    coeffs[:, i_point] = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(jacobian.T, jacobian)/self.measurement_variance+numpy.dot(jacobian_int.T, jacobian_int)/self.intrinsic_variance), jacobian.T), noisy_sensor_step[:, i_point]-noisy_sensor_base[:, i_point])/self.measurement_variance
            current_cost = train(noisy_sensor_base[:, point_in_batch].reshape((self.dim_measurements, n_batch)).T,  coeffs[:, point_in_batch].reshape((self.dim_intrinsic, n_batch)).T)
            if iteration % 100 == 0:
                for i_point in numpy.random.choice(n_valid_points, size=(n_valid_points), replace=False):
                    jacobian = self.get_jacobian_val(noisy_sensor_valid_base[:, i_point].reshape((self.dim_measurements, 1)))[0,:,:]
                    jacobian_int = self.get_jacobian_int_val(noisy_sensor_valid_base[:, i_point].reshape((self.dim_measurements, 1)))[0,:,:]
                    coeffs_valid[:, i_point] = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(jacobian.T, jacobian)/self.measurement_variance+numpy.dot(jacobian_int.T, jacobian_int)/self.intrinsic_variance), jacobian.T), noisy_sensor_valid_step[:, i_point]-noisy_sensor_valid_base[:, i_point])/self.measurement_variance
            current_valid_cost = train_valid(noisy_sensor_valid_base[:, :].reshape((self.dim_measurements, n_valid_points)).T, coeffs_valid[:, :].reshape((self.dim_intrinsic, n_valid_points)).T)

            iteration += 1

            print("iteration=", iteration, "learning_rate=", learning_rate.get_value(), "cost:", current_cost.mean(), "cost valid:", current_valid_cost.mean())

            cost_term_2.append(current_cost.mean())
            cost_term_valid_2.append(current_valid_cost.mean())

            if iteration % 50 == 0:
                learning_rate.set_value(0.99*learning_rate.get_value())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cost_term_2[5:], c='b')
        ax.plot(cost_term_valid_2[5:], c='r')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Cost')
        ax.set_title("Cost vs Epochs - Stage II")