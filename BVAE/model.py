import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer


class VAE:
    def __init__(self, args, q_dims=None):
        usersNum, itemsNum = args.user_num + 1, args.item_num + 1
        # p:encoder, h:hidden, q:decoder
        if type(args.hiddenDim) == list:
            p_dims = args.hiddenDim
            p_dims.append(itemsNum)
        else:
            p_dims = []
            p_dims.append(args.hiddenDim)
            p_dims.append(itemsNum)
        self.p_dims = p_dims
        if q_dims is None:
            # ::-1 倒序输出
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims

        self.dims = self.q_dims + self.p_dims[1:]
        self.h_dims = args.hiddenDim
        self.num_behaviors = args.behavior_num
        self.num_experts = args.specific_expert_num

        self.lam = args.reg_scale
        self.lr = args.lr_rate
        self.random_seed = args.random_seed
        self.optimizer = args.optimizer

    def create_placeholders(self):

        self.input_ph_union = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])

        self.input_ph_behavior1 = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.input_ph_behavior2 = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        if self.num_behaviors >= 3:
            self.input_ph_behavior3 = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
            if self.num_behaviors >= 4:
                self.input_ph_behavior4 = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])

        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)

        self.prediction_top_k = tf.placeholder(tf.float32, [None, None])
        self.scale_top_k = tf.placeholder(tf.int32)
        self.top_k = tf.nn.top_k(self.prediction_top_k, self.scale_top_k)

    def create_params(self):

        self.weights_q_mu, self.biases_q_mu = [], []
        self.weights_q_std, self.biases_q_std = [], []
        self.weights_p, self.biases_p = [], []
        self.weights_g, self.biases_g = [], []
        self.weights_g_2, self.biases_g_2 = [], []

        init_w = tf.contrib.layers.xavier_initializer(seed=self.random_seed)
        init_b = tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)

        # params of q network
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):

            shape_w = [d_in, d_out]
            shape_b = [d_out]
            # behaviors: 0 for union, 1, 2, 3, 4 for behaviors
            for j in range(self.num_behaviors + 1):
                name_w = 'weights_q_behavior{}'.format(j)
                name_b = 'biases_q_behavior{}'.format(j)
                mu_weights, mu_biases, std_weights, std_biases = [], [], [], []
                mu_weights.append(tf.get_variable(name=name_w + '_mu', shape=shape_w, initializer=init_w))
                mu_biases.append(tf.get_variable(name=name_b + '_mu', shape=shape_b, initializer=init_b))
                std_weights.append(tf.get_variable(name=name_w + '_std', shape=shape_w, initializer=init_w))
                std_biases.append(tf.get_variable(name=name_b + '_std', shape=shape_b, initializer=init_b))
                self.weights_q_mu.append(mu_weights)
                self.biases_q_mu.append(mu_biases)
                self.weights_q_std.append(std_weights)
                self.biases_q_std.append(std_biases)

        d_hid = self.h_dims
        # params of p network
        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            # shape_w = [d_in + d_hid, d_out]
            shape_w = [d_hid, d_out]
            shape_b = [d_out]
            # tasks: 0, 1, 2, 3
            for j in range(self.num_behaviors):
                name_w = 'weight_q_behavior{}'.format(j + 1)
                name_b = 'biases_q_behavior{}'.format(j + 1)
                network_weights, network_biases = [], []
                network_weights.append(tf.get_variable(name=name_w, shape=shape_w, initializer=init_w))
                network_biases.append(tf.get_variable(name=name_b, shape=shape_b, initializer=init_b))
                self.weights_p.append(network_weights)
                self.biases_p.append(network_biases)

        # params of Gating
        std_g = tf.sqrt(tf.div(2.0, d_hid + 1))
        for j in range(self.num_behaviors):
            W_g1 = tf.Variable(tf.truncated_normal(shape=[d_hid, 1], mean=0.0, stddev=std_g),
                               name='Weights_for_Gating1_{}'.format(j + 1), dtype=tf.float32)
            b_g1 = tf.Variable(tf.truncated_normal(shape=[1, 1], mean=0.0, stddev=std_g),
                               name='Bias_for_Gating1_{}'.format(j + 1), dtype=tf.float32)
            W_g2 = tf.Variable(tf.truncated_normal(shape=[d_hid, 1], mean=0.0, stddev=std_g),
                               name='Weights_for_Gating2_{}'.format(j + 1), dtype=tf.float32)
            b_g2 = tf.Variable(tf.truncated_normal(shape=[1, 1], mean=0.0, stddev=std_g),
                               name='Bias_for_Gating2_{}'.format(j + 1), dtype=tf.float32)
            self.weights_g.append(W_g1)
            self.biases_g.append(b_g1)
            self.weights_g_2.append(W_g2)
            self.biases_g_2.append(b_g2)

    def q_get_mu(self):

        input_data, mu_q = self.input_ph_union, None

        w_q = self.weights_q_mu[0]
        b_q = self.biases_q_mu[0]

        h = tf.nn.l2_normalize(input_data, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(w_q, b_q)):
            h = tf.matmul(h, w) + b
            if i != len(w_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h
        return mu_q

    def q_get_logvar(self, behavior_type):
        input_data, logvar_q = [], None
        if behavior_type == 0:
            input_data = self.input_ph_union
        if behavior_type == 1:
            input_data = self.input_ph_behavior1
        if behavior_type == 2:
            input_data = self.input_ph_behavior2
        if behavior_type == 3:
            input_data = self.input_ph_behavior3
        if behavior_type == 4:
            input_data = self.input_ph_behavior4

        w_q = self.weights_q_std[behavior_type]
        b_q = self.biases_q_std[behavior_type]

        h = tf.nn.l2_normalize(input_data, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(w_q, b_q)):
            h = tf.matmul(h, w) + b
            if i != len(w_q) - 1:
                h = tf.nn.tanh(h)
            else:
                logvar_q = h
        return logvar_q

    def g_mu(self, behavior_type):

        w_g = self.weights_g[behavior_type - 1]
        b_g = self.biases_g[behavior_type - 1]

        # mu_q_target = self.q_get_mu(behavior_type=behavior_type)
        mu_q_target = self.q_get_mu()
        weight_target = tf.nn.softmax(tf.matmul(mu_q_target, w_g) + b_g)
        mu_q = mu_q_target * weight_target

        # for i in range(self.num_behaviors + 1):
        #     if i != behavior_type:
        #         mu_q_i = self.q_get_mu(behavior_type=0)
        #         weights = tf.nn.softmax(tf.matmul(mu_q_i, w_g) + b_g)
        #         mu_q += mu_q_i * weights
        return mu_q

    def g_z(self, behavior_type):

        self.KL_target, self.tem, self.sampled_z = [], [], []
        for i in range(self.num_behaviors):
            mu_q_i = self.g_mu(i + 1)
            logvar_q_i = self.q_get_logvar(i + 1)
            std_q_i = tf.exp(0.5 * logvar_q_i)

            epsilon_i = tf.random_normal(tf.shape(std_q_i))
            sampled_z_behavior_i = mu_q_i + self.is_training_ph * epsilon_i * std_q_i
            self.sampled_z.append(sampled_z_behavior_i)

            kl_i = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar_q_i + tf.exp(logvar_q_i) + mu_q_i ** 2 - 1), axis=1))
            self.KL_target.append(kl_i)

            tem_i = tf.square(tf.reduce_mean(std_q_i, 1))
            self.tem.append(tem_i)

        w_g = self.weights_g_2[behavior_type - 1]
        b_g = self.biases_g_2[behavior_type - 1]

        z_target = self.sampled_z[behavior_type - 1]
        weight_target = tf.nn.softmax(tf.matmul(z_target, w_g) + b_g)
        z = z_target * weight_target

        for i in range(1, self.num_behaviors + 1):
            if i != behavior_type:
                z_i = self.sampled_z[i - 1]
                weights = tf.nn.softmax(tf.matmul(z_i, w_g) + b_g)
                z += z_i * weights
        return z

    def p_graph(self, behavior_type):

        w_p = self.weights_p[behavior_type - 1]
        b_p = self.biases_p[behavior_type - 1]

        h = self.g_z(behavior_type=behavior_type)
        for i, (w, b) in enumerate(zip(w_p, b_p)):
            h = tf.matmul(h, w) + b
            if i != len(w_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def inference(self):

        self.logits_target = []
        for i in range(self.num_behaviors):
            self.logits_target.append(self.p_graph(i))

    def create_loss(self):

        neg_ELBO_KL = 0.
        input_ph_behavior = None
        for i in range(self.num_behaviors):
            if i == 0:
                input_ph_behavior = self.input_ph_behavior1
            elif i == 1:
                input_ph_behavior = self.input_ph_behavior2
            elif i == 2:
                input_ph_behavior = self.input_ph_behavior3
            elif i == 3:
                input_ph_behavior = self.input_ph_behavior4

            log_softmax_var = tf.nn.log_softmax(self.logits_target[i])
            tem = self.tem[i]
            neg_ll_target = - (0.5 / tem) * tf.reduce_sum(log_softmax_var * input_ph_behavior, axis=-1)

            neg_ELBO_KL += tf.reduce_mean(neg_ll_target + tf.log(tem)) + self.anneal_ph * self.KL_target[i]

        reg_var = 0.
        reg = l2_regularizer(self.lam)
        for i in range(self.num_behaviors):
            reg_var += apply_regularization(reg, self.weights_q_mu[i] + self.weights_p[i])

        self.loss = reg_var + neg_ELBO_KL

    def create_optimizer(self):
        if self.optimizer == 'GD':
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss=self.loss, var_list=self.params)
        elif self.optimizer == 'Adam':
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        elif self.optimizer == 'Adagrad':
            self.train_op = tf.train.AdagradOptimizer(self.lr).minimize(loss=self.loss, var_list=self.params)

    def build_graph(self):
        self.create_params()
        self.create_placeholders()
        self.inference()
        self.create_loss()
        self.create_optimizer()
