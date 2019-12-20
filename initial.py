import numpy as np
import tensorflow as tf
import os


# embedding the position 嵌入位置



def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


# reading data
def init():
    print('reading word embedding data...')  #读vec.txtx里面的数据
    vec = []   #装vec.txt中每行后面的数字
    word2id = {}   # key为vec.txt中每行第一个字  value值为从0开始的下标
    f = open('./origin_data/vec.txt', encoding='utf-8')
    content = f.readline()
    content = content.strip().split()
    dim = int(content[1])  #向量维度 结果为100
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [(float)(i) for i in content]  #将vec.txt每行中的内容 从第2个位置开始后的所有内容 变成一个float型数组
                                                #比如[-0.05855, 0.125846, -0.083195, 0.031818, -0.183519,...]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)  #至此 vec.txt中所有的词对应的key value完成 如{',': 0, '的': 1, '。': 2, '1': 3, '0': 4, '年': 5, '、': 6, '一': 7,...}

    # np.random.normal返回一个由size指定形状的数组，数组中的值服从 μ=loc,σ=scale 的正态分布。
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

##########读vec.txt中的数据结束   vec是后面所有数字组成的数组，word2id是所有的词和下标值组成的字典

    print('reading relation to id')
    relation2id = {}    #key值为relation2id.txt中的第一列  value值为对应的第二列
    f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

    # length of sentence is 70
    fixlen = 70
    # max length of position embedding is 60 (-60~+60)
    maxlen = 60
    #train_sen存的是实体对和字嵌入向量和关系触发词向量  train_ans存的是实体对和关系向量
    train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector

    print('reading train data...')
    f = open('./origin_data/train.txt', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split()
        # get entity name
        en1 = content[0]
        en2 = content[1]
        relation = 0
        if content[2] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[2]]
        # put the same entity pair sentences into a dict
        tup = (en1, en2)
        label_tag = 0  #表示实体对和关系数组匹配的所在位置
        if tup not in train_sen:
            train_sen[tup] = []  #结果为：如{('龋', 'X线牙片'): []}
            train_sen[tup].append([]) #结果为 ：如{('龋', 'X线牙片'): [[]]}
            y_id = relation  #y_id为关系的id
            label_tag = 0
            label = [0 for i in range(len(relation2id))]  #结果为[0, 0, 0, 0]
            label[y_id] = 1 #表示 当前句的关系是什么 比如[0,0,1,0]  表示关系为Related-CH
            train_ans[tup] = []  #结果为：如{('龋', 'X线牙片'): []}
            train_ans[tup].append(label) #结果为：如{('龋', 'X线牙片'): [[0, 0, 1, 0]]}
        else:
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1

            temp = find_index(label, train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                label_tag = len(train_ans[tup]) - 1  #label_tag表示添加的label所处list中的index
                train_sen[tup].append([])            #train_sen再添加一层[]
            else:
                label_tag = temp

        sentence = content[3]
        # relation_word = content[4]

        en1pos = 0
        en2pos = 0

        #For Chinese
        en1pos = sentence.find(en1) #找到实体在句子中的位置
        if en1pos == -1:
            en1pos = 0
        en2pos = sentence.find(en2)
        if en2pos == -1:
            en2post = 0
        
        output = []

        #Embeding the position
        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])  #存的是位置从0到69的字嵌入位置 [[16116, 58, 47], [16116, 59, 48], [16116, 60, 49], [16116, 61, 50],...]]
            # print(output)
            # print(relation_word_att)
            # output = tf.concat([output,relation_word_att],0)
            # print(output)
        # Embeding word2id


        for i in range(min(fixlen, len(sentence))):
            word = 0
            # 找句子中的每一字在字向量表中的位置
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            output[i][0] = word  #存的是句子的词嵌入向量
        # print('增加关系发现词前output',output)  #保存每句话中每个字的id和距离实体位置  长度为70  多出的部分字id默认为16116
        # train_sen[tup][label_tag].append(output)
        # print('增加关系发现词前train_sen', train_sen)
        # relation_word_att = []
        # for i in range(len(relation_word)):
        #     if relation_word[i] not in word2id:
        #         q = word2id['UNK']
        #     else:
        #         q = word2id[relation_word[i]]
        #     relation_word_att.append(q)
        # output.append(relation_word_att)
        # print('增加关系发现词后output', output)
        train_sen[tup][label_tag].append(output)
        # print('增加关系发现词后train_sen', train_sen)  #至此 已将关系发现词的id数组加到了 train_sen的最后
        # 增加关系发现词后train_sen {('骨髓瘤', '骨质流失'): [[[[66, 60, 54], [1108, 61, 55], [3019, 62, 56], [2347, 63, 57], [454, 64, 58], [237, 65, 59],...,[454, 237]]]]}
        # exit()
    print('reading test data ...')

    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)

    f = open('./origin_data/test.txt', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split()
        en1 = content[0]
        en2 = content[1]
        relation = 0
        if content[2] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[2]]
        tup = (en1, en2)

        if tup not in test_sen:
            test_sen[tup] = []
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            test_ans[tup] = label
        else:
            y_id = relation
            test_ans[tup][y_id] = 1

        sentence = content[3]

        en1pos = 0
        en2pos = 0
        
        #For Chinese
        en1pos = sentence.find(en1)
        if en1pos == -1:
            en1pos = 0
        en2pos = sentence.find(en2)
        if en2pos == -1:
            en2post = 0
            
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word
        test_sen[tup].append(output)



    train_x = []  #把字典进行列表表示 train_x存的是训练集的词嵌入向量  train_y存的是关系向量
    train_y = []
    test_x = []
    test_y = []

    print('organizing train data')
    f = open('./data/train_q&a.txt', 'w', encoding='utf-8')
    temp = 0
    for i in train_sen:
#i为实体对
        if len(train_ans[i]) != len(train_sen[i]):
            print('ERROR')
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    # print(train_x)
    # exit()
    f.close()

    print('organizing test data')
    f = open('./data/test_q&a.txt', 'w', encoding='utf-8')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    f.close()

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    np.save('./data/vec.npy', vec)
    np.save('./data/train_x.npy', train_x)
    np.save('./data/train_y.npy', train_y)
    np.save('./data/testall_x.npy', test_x)
    np.save('./data/testall_y.npy', test_y)





def seperate():
    print('reading training data')
    x_train = np.load('./data/train_x.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []

    print('seprating train data')
    for i in range(len(x_train)): #每个实体对
        word = []
        pos1 = []
        pos2 = []
        for j in x_train[i]: #每个实体对下面的句子
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:   #每个实体对下面的每个单词
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    print(len(train_word))
    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save('./data/train_word.npy', train_word)
    np.save('./data/train_pos1.npy', train_pos1)
    np.save('./data/train_pos2.npy', train_pos2)

    print('seperating test all data')
    x_test = np.load('./data/testall_x.npy')
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)
    print(len(test_word))

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./data/testall_word.npy', test_word)
    np.save('./data/testall_pos1.npy', test_pos1)
    np.save('./data/testall_pos2.npy', test_pos2)



# get answer metric for PR curve evaluation
def getans():
    test_y = np.load('./data/testall_y.npy')
    eval_y = []
    for i in test_y:# #遍历列表
        eval_y.append(i[1:])   #i[1:]得到的结果是每一个关系向量中的后三个数
    allans = np.reshape(eval_y, (-1)) #"将eval_y数组展成一行"  存的是测试集中每一个实体对关系向量中的后面三个数
    np.save('./data/allans.npy', allans)

#从vec得到原始文件
def get_metadata(): #逐行写入
    fwrite = open('./data/metadata.tsv', 'w', encoding='utf-8')
    f = open('./origin_data/vec.txt', encoding='utf-8')
    f.readline()
    while True:
        content = f.readline().strip()
        if content == '':
            break
        name = content.split()[0]  #得到 字
        fwrite.write(name + '\n')
    f.close()
    fwrite.close()


init()
seperate()
getans()
get_metadata()

#train_sen 保存为{（e1,e2）:[字id,pos1,pos2]}
# train_ans保存为{e1,e2）:关系向量数组}
# train_x保存为字嵌入向量数组
# train_y保存为每个实体对的关系向量数组
# train_word为字的id
# train_pos1为每个字与第一个实体的距离
# train_pos2为每个字与第二个实体的距离
# allans.npy保存的是测试集中所有关系数组中的后三个数字所组成的数组