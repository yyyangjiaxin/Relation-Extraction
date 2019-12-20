import tensorflow as tf
import numpy as np


class Settings(object):
    def __init__(self):
        self.vocab_size = 16691    #词典的规模 总共16691个词
        self.num_steps = 70       #句子序列最大长度
        self.num_epochs = 20      #表示批次总数，也就是说，需要向session喂这么多批数据  批处理的句子数是10
        self.num_classes = 3     # 关系数目
        self.gru_size = 230        #隐藏层神经元的个数，GRU网络单元的个数，也即隐藏层的节点数。这个参数表示的是用于记忆和储存过去状态的节点个数。
        # self.keep_prob = 0.5      #用于dropout.每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作，可以防止过拟合
        self.keep_prob = 1
        self.num_layers = 1       #GRU的层数
        self.pos_size = 5
        self.pos_num = 123     #0,X+61,122 共123个
        self.big_num = 50        # the number of entity pairs of each batch during training or testing #在训练或测试期间每个批次的实体对数目
        # self.big_num = 33


class GRU:
    def __init__(self, is_training, word_embeddings, settings):
        self.num_steps = num_steps = settings.num_steps
        self.vocab_size = vocab_size = settings.vocab_size
        self.num_classes = num_classes = settings.num_classes
        self.gru_size = gru_size = settings.gru_size
        self.big_num = big_num = settings.big_num

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_word')#此函数可以理解为形参，插入一个张量的占位符，这个张量总是被喂入图片数据。用于定义过程，在执行的时候再赋具体的值,返回：Tensor 类型.参数dtype：数据类型。常用的是tf.float32,tf.float64等数值类型shape：数据维度。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定;name：数据名称。
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos2')
        # self.input_rel = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_rel')
        self.input_rel = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_rel')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[big_num + 1], name='total_shape')
        total_num = self.total_shape[-1]

        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')   #initializer变量初始化的方式，name变量名称    取值
        pos1_embedding = tf.get_variable('pos1_embedding', [settings.pos_num, settings.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [settings.pos_num, settings.pos_size])
        rel_embedding = tf.get_variable(initializer=word_embeddings, name='rel_embedding')
        attention_w = tf.get_variable('attention_omega', [gru_size, 1])
        sen_a = tf.get_variable('attention_A', [gru_size])
        sen_r = tf.get_variable('query_r', [gru_size, 1])
        relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, gru_size])
        sen_d = tf.get_variable('bias_d', [self.num_classes])

        gru_cell_forward = tf.contrib.rnn.GRUCell(gru_size)
        gru_cell_backward = tf.contrib.rnn.GRUCell(gru_size)

        if is_training and settings.keep_prob < 1:
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=settings.keep_prob)    #所谓dropout,就是指网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值。这是是一种有效的正则化方法，可以有效防止过拟合。
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=settings.keep_prob)

        cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * settings.num_layers)
        cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * settings.num_layers)

        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []
        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0
        # 参数初始化,rnn_cell.RNNCell.zero_stat
        self._initial_state_forward = cell_forward.zero_state(total_num, tf.float32)
        self._initial_state_backward = cell_backward.zero_state(total_num, tf.float32)

        # embedding layer
        # inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(word_embedding, self.input_word),  #tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素，，＃找到要寻找的embedding data中的对应的行下的vector。
        #                                            tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
        #                                            tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)])
        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(word_embedding, self.input_word),# tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素，，＃找到要寻找的embedding data中的对应的行下的vector。
                                                   tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
                                                   tf.nn.embedding_lookup(pos2_embedding, self.input_pos2),
                                                   tf.nn.embedding_lookup(rel_embedding, self.input_rel)])

        # inputs_backward = tf.concat(axis=2,
        #                             values=[tf.nn.embedding_lookup(word_embedding, tf.reverse(self.input_word, [1])),
        #                                     tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1, [1])),
        #                                     tf.nn.embedding_lookup(pos2_embedding, tf.reverse(self.input_pos2, [1]))])
        inputs_backward = tf.concat(axis=2,
                                    values=[tf.nn.embedding_lookup(word_embedding, tf.reverse(self.input_word, [1])),
                                            tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1, [1])),
                                            tf.nn.embedding_lookup(pos2_embedding, tf.reverse(self.input_pos2, [1])),
                                            tf.nn.embedding_lookup(rel_embedding, tf.reverse(self.input_rel, [1]))])
        outputs_forward = []

        state_forward = self._initial_state_forward

        # Bi-GRU layer
        with tf.variable_scope('GRU_FORWARD') as scope:
            for step in range(num_steps):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_forward, state_forward) = cell_forward(inputs_forward[:, step, :], state_forward)
                outputs_forward.append(cell_output_forward)

        outputs_backward = []

        state_backward = self._initial_state_backward
        with tf.variable_scope('GRU_BACKWARD') as scope:
            for step in range(num_steps):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_backward, state_backward) = cell_backward(inputs_backward[:, step, :], state_backward)
                outputs_backward.append(cell_output_backward)

        output_forward = tf.reshape(tf.concat(axis=1, values=outputs_forward), [total_num, num_steps, gru_size])
        output_backward = tf.reverse(
            tf.reshape(tf.concat(axis=1, values=outputs_backward), [total_num, num_steps, gru_size]),
            [1])

        # word-level attention layer
        output_h = tf.add(output_forward, output_backward)
        attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [total_num * num_steps, gru_size]), attention_w),
                       [total_num, num_steps])), [total_num, 1, num_steps]), output_h), [total_num, gru_size])

        # sentence-level attention layer
        for i in range(big_num):

            sen_repre.append(tf.tanh(attention_r[self.total_shape[i]:self.total_shape[i + 1]]))
            batch_size = self.total_shape[i + 1] - self.total_shape[i]

            sen_alpha.append(
                tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre[i], sen_a), sen_r), [batch_size])),
                           [1, batch_size]))

            sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [gru_size, 1]))
            sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s[i]), [self.num_classes]), sen_d))#multiply这个函数实现的是元素级别的相乘，也就是两个相乘的数元素各自相乘，

            self.prob.append(tf.nn.softmax(sen_out[i]))
            with tf.name_scope("output"):
                self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

            with tf.name_scope("loss"):
                self.loss.append(
                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.input_y[i])))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]

            # tf.summary.scalar('loss',self.total_loss)
            # tf.scalar_summary(['loss'],[self.total_loss])
            with tf.name_scope("accuracy"):
                self.accuracy.append(
                    tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),
                                   name="accuracy"))

        # tf.summary.scalar('loss',self.total_loss)
        tf.summary.scalar('loss', self.total_loss)
        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),#应用正则化方法到参数上，regularizer:就是我们上一步创建的正则化方法,
                                                              weights_list=tf.trainable_variables())   #weights_list: 想要执行正则化方法的参数列表,如果为None的话,就取GraphKeys.WEIGHTS中的weights.函数返回一个标量Tensor,同时,这个标量Tensor也会保存到GraphKeys.REGULARIZATION_LOSSES中.这个Tensor保存了计算正则项损失的方法.
        self.final_loss = self.total_loss + self.l2_loss
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('final_loss', self.final_loss)
