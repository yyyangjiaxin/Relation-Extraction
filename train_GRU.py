import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from tensorflow.contrib.tensorboard.plugins import projector
tf.set_random_seed(1)
np.random.seed(1)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')
# 设置log输出信息的 默认值是1 显示所有信息，2是显示warning和Error  3是只显示Error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(_):
    # the path to save models
    save_path = './model/'

    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')   #字向量表
    # print("wordembedding",wordembedding)
    # print('len(wordembedding)',len(wordembedding))

    print('reading training data')
    train_y = np.load('./data/train_y.npy')   #分类标签数组  即训练集关系向量数组
    # print('train_y',train_y)
    train_word = np.load('./data/train_word.npy')  #每句话种每个字的id数组
    # print('train_word', train_word)
    train_pos1 = np.load('./data/train_pos1.npy')   #第一个实体嵌入位置数组
    # print('train_pos1', train_pos1)
    train_pos2 = np.load('./data/train_pos2.npy')  #第二个实体嵌入位置数组
    # print('train_pos2', train_pos2)
    # train_rel = np.load('./data/train_rel.npy')
    # print('train_rel',train_rel)

    settings = network.Settings()   #调用Seetings设置变量
    settings.vocab_size = len(wordembedding)     #结果为16117
    settings.num_classes = len(train_y[0])       #结果为4


    big_num = settings.big_num
    keep_prob = settings.keep_prob
    print('keep_prob',keep_prob)

    with tf.Graph().as_default():   #tf.Graph() 表示实例化了一个类,tf.Graph().as_default() 表示将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图，

        sess = tf.Session()  #运行Tensorflow操作的类, tf.Session()：创建一个会话, tf.Session().as_default()创建一个默认会话
        with sess.as_default():   #执行操作

            initializer = tf.contrib.layers.xavier_initializer()  #初始化权重矩阵
            with tf.variable_scope("model", reuse=None, initializer=initializer):  #tf.variable_scope用于定义创建变量(层)的操作的上下文管理器. reuse=None继承父范围的重用标志
                m = network.GRU(is_training=True, word_embeddings=wordembedding, settings=settings)    #调用GRU
            global_step = tf.Variable(0, name="global_step", trainable=False)   #定义全局步骤变量  不可训练
            optimizer = tf.train.AdamOptimizer(0.0005)    #优化算法  优化器optimizer

            train_op = optimizer.minimize(m.final_loss, global_step=global_step) #添加操作节点，用于最小化loss
            sess.run(tf.global_variables_initializer())  #初始化模型的参数
            saver = tf.train.Saver(max_to_keep=None)   #实例化对象 用于保存模型 max_to_keep的值为None表示保存所有的checkpoint文件，默认值为5

            merged_summary = tf.summary.merge_all()   #用于管理summary  将所有summary全部保存到磁盘 以便tensorboard显示
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)  #指定一个文件用来保存图

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, big_num):  #训练每一步
                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                # total_rel = []
                # print(word_batch)  50条句子
                for i in range(len(word_batch)):
                    # print(len(word_batch))  结果为1
                    total_shape.append(total_num)

                    total_num += len(word_batch[i])
                    for word in word_batch[i]:  #向下取一层
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                    # for rel in rel_batch[i]:
                    #     total_rel.append(rel)
                total_shape.append(total_num)
                # print(total_num)
                # print(total_shape)
                # print(len(total_shape))
                # print(total_word)
                # print(len(total_word))
                # print(total_pos1)
                # print(len(total_pos1))
                # print(total_pos2)
                # print(len(total_pos2))
                # exit()
                # for i in range(len(total_rel)):
                #     # a = total_rel[i]
                #     print(type(total_rel))
                #     print(type(total_rel[0]))
                #     print(len(total_rel[i]))
                #     while len(total_rel[i]) !=70:
                #         total_rel[i].append(16116)
                total_shape = np.array(total_shape)   #[0,1,2,3...50]
                total_word = np.array(total_word)  #50条句子里 每条句子中70个字中 每个字的id 长度为50
                total_pos1 = np.array(total_pos1)  #50条句子里 每条句子中70个字中 每个字到第一个实体位置的距离 长度为50
                total_pos2 = np.array(total_pos2)  #50条句子里 每条句子中70个字中 每个字到第二个实体位置的距离 长度为50
                # total_rel = np.array(total_rel)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch   #50条句子的关系向量数组
                # feed_dict[m.input_rel] = total_rel
                # print(total_shape)
                # print(len(total_shape))
                # print(total_word)
                # print(len(total_word[0]))
                # print(total_pos1)
                # print(len(total_pos1[0]))
                # print(y_batch)
                # print(len(y_batch))
                # print(total_rel)
                # print(len(total_rel[0]))
                # exit()

                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)
                # print('step',step)
                if step % 50 == 0:
                #if step % 10 == 0:
                    tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                    print(tempstr)

            for one_epoch in range(settings.num_epochs):
                # print('one_epoch',one_epoch)
                temp_order = list(range(len(train_word)))  #train_word中存的是每个句子中每个字的id数组，长度是句子总数{0,1....866}
                # print('temp_order',temp_order)
                np.random.shuffle(temp_order)   #打乱顺序函数
                # print('temp_order', temp_order)
                # print('len(temp_order)',len(temp_order))
                for i in range(int(len(temp_order) / float(settings.big_num))): #每次最多丢进去50个实体对 一共需要丢多少次
                    # print('i',i)
                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []
                    # temp_rel = []

                    temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                    # print('temp_input',temp_input)
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])   #关系向量数组
                        # temp_rel.append(train_rel[k])
                    num = 0
                    for single_word in temp_word:
                        # print(len(single_word[0]))  结果为70
                        num += len(single_word)

                    if num > 1500:
                        print('out of range')
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)
                    # temp_rel = np.array(temp_rel)

                    train_step(temp_word, temp_pos1, temp_pos2, temp_y,settings.big_num)

                    current_step = tf.train.global_step(sess, global_step)  #global_step代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表。
                    #if current_step > 8000 and current_step % 100 == 0:
                    # print('current_step',current_step)
                    # if current_step > 80 and current_step % 5== 0:
                    if current_step > 300 and current_step % 10 == 0:
                        # print('saving model')
                        path = saver.save(sess, save_path + 'ATT_GRU_model', global_step=current_step)
                        tempstr = 'have saved model to ' + path
                        print(tempstr)


if __name__ == "__main__":
    tf.app.run()
