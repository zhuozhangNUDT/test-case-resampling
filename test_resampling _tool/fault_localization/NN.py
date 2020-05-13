#-*- coding:utf-8 -*-#
# Copyright (c) National University of Defense Technology(NUDT).
# All rights reserved.
#
"""
Created on 2020-05-11

@author: zhangzhuo

usage : 
    python NN.py dev
    or
    python NN.py dev_resampling
"""
import sys
from configparser import ConfigParser
import logging.config
import tensorflow as tf
import numpy as np
from scipy.sparse import dok_matrix
import math

# initialize logger
logging.config.fileConfig("logging.cfg")
logger = logging.getLogger("root")

class RuntimeContext(object):
    """ runtime enviroment
    """
    
    def __init__(self):
        """ initialization
        """
        # configuration initialization
        config_parser = ConfigParser()
        config_file = self.get_config_file_name()
        config_parser.read(config_file, encoding="UTF-8")
        sections = config_parser.sections()
        
        coverage_information_matrix_section = sections[0]
        self.covMatrix = config_parser.get(coverage_information_matrix_section, "covMatrix")
        
        test_cases_results_section = sections[1]
        self.error = config_parser.get(test_cases_results_section, "error")
        
        DL_result_section = sections[2]
        self.DL_result = config_parser.get(DL_result_section, "DL_result")
        
        parameters = sections[3]
        self.learning_rate = config_parser.get(parameters, "learning_rate")
        self.drop = config_parser.get(parameters, "drop")
        self.epochs_drop = config_parser.get(parameters, "epochs_drop")
        self.batch_size = config_parser.get(parameters, "batch_size")
        self.in_units = config_parser.get(parameters, "in_units")
        self.test_num = config_parser.get(parameters, "test_num")
        
    def get_config_file_name(self):
        """ get the configuration file name according to the command line parameters
        """
        argv = sys.argv
        config_type = "dev" # default configuration type
        if None != argv and len(argv) > 1 :
            config_type = argv[1]
        config_file = config_type + ".cfg"
        logger.info("get_config_file_name() return : " + config_file)
        return config_file
    
def main():
    runtime_context = RuntimeContext()    
    f1 = open(runtime_context.covMatrix,'r')
    f2 = open(runtime_context.error,'r')
    f3 = open(runtime_context.DL_result,'w')

    first_ele = True
    for data in f1.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_x = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_x = np.c_[matrix_x,nums]
    f1.close()
    
    first_ele = True
    for data in f2.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_y = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_y = np.c_[matrix_y,nums]
    f2.close()
    
    
    matrix_x = matrix_x.transpose()
    matrix_y = matrix_y.transpose()

    # Parameters
    learning_rate = float(runtime_context.learning_rate)
    drop = float(runtime_context.drop)
    epochs_drop = float(runtime_context.epochs_drop)
    lr= tf.placeholder(tf.float32)
    batch_size = int(runtime_context.batch_size)
    epoch_loss = 0
    # Network Parameters cuda
    in_units = int(runtime_context.in_units)     # coverage data input
    #test_num = 1510    # test case number
    test_num = int(runtime_context.test_num)    # test case number4070
    hidden_units_num = int(round(in_units/300)*10)
    h1_units = hidden_units_num    # input layer num of features
    h2_units = hidden_units_num    # hidden layer num of features
    h3_units = hidden_units_num    # hidden layer num of features
    h4_units = hidden_units_num    # hidden layer num of features
    out_units = 1    # output layer num of features
    
    x = tf.placeholder(tf.float32,[None,in_units])
    y_ = tf.placeholder(tf.float32,[batch_size,out_units])
    keep_prob = tf.placeholder(tf.float32)
    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev = 0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    
    W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev = 0.1))
    b2 = tf.Variable(tf.zeros([h2_units]))
    
    W3 = tf.Variable(tf.truncated_normal([h2_units, h3_units], stddev = 0.1))
    b3 = tf.Variable(tf.zeros([h3_units]))
    
    W4 = tf.Variable(tf.truncated_normal([h3_units, h4_units], stddev = 0.1))
    b4 = tf.Variable(tf.zeros([h4_units]))
    
    W5 = tf.Variable(tf.zeros([h4_units, out_units]))
    b5 = tf.Variable(tf.zeros([out_units]))
    
    hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    hidden2 = tf.nn.relu(tf.matmul(hidden1_drop,W2) + b2)
    hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
    hidden3 = tf.nn.relu(tf.matmul(hidden2_drop,W3) + b3)
    hidden3_drop = tf.nn.dropout(hidden3, keep_prob)
    hidden4 = tf.nn.relu(tf.matmul(hidden3_drop,W4) + b4)
    hidden4_drop = tf.nn.dropout(hidden4, keep_prob)
    y = tf.nn.sigmoid(tf.matmul(hidden4_drop,W5)+b5)
    
    loss = tf.reduce_mean(tf.square(y-y_))
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    init = tf.initialize_all_variables()
    
    
    #run session
    sess = tf.Session()
    sess.run(init)
    
    for step in range(0,test_num * 300,batch_size):
    
        sess.run(train_step,feed_dict={lr:learning_rate,x:np.mat(matrix_x[step%test_num:step%test_num+batch_size,]),y_:np.mat(matrix_y[step%test_num:step%test_num+batch_size,]), keep_prob: 0.75})
        epoch = math.floor(step/test_num)
        epoch_loss = epoch_loss+sess.run(loss,feed_dict={lr:learning_rate,x:np.mat(matrix_x[step%test_num:step%test_num+batch_size,]),y_:np.mat(matrix_y[step%test_num:step%test_num+batch_size,]), keep_prob: 0.75}) 

        if step%test_num == 0:# and step > 0:
            print ("epoch:"+str(epoch), "learning_rate:"+str(learning_rate) , "loss:"+str(epoch_loss*batch_size/test_num))
            epoch_loss = 0
            if epoch%10 == 0:
                learning_rate = learning_rate * math.pow(drop,math.floor((epoch)/epochs_drop))                
       
    #initialize test
    in_units1 = 1
    test = tf.placeholder(tf.float32,[in_units1,in_units])
       
    hidden1 = tf.nn.relu(tf.matmul(test,W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    hidden2 = tf.nn.relu(tf.matmul(hidden1_drop,W2) + b2)
    hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
    hidden3 = tf.nn.relu(tf.matmul(hidden2_drop,W3) + b3)
    hidden3_drop = tf.nn.dropout(hidden3, keep_prob)
    hidden4 = tf.nn.relu(tf.matmul(hidden3_drop,W4) + b4)
    hidden4_drop = tf.nn.dropout(hidden4, keep_prob)
    result = tf.nn.sigmoid(tf.matmul(hidden4_drop,W5)+b5)
    
    #initialize matrix
    for j in range(in_units):
       S = dok_matrix((in_units1,in_units),dtype = np.float32)
       S[0,j] = 1
       matrix_test = S.toarray()
       f3.write(str(sess.run(result,feed_dict={test:matrix_test, keep_prob: 0.75})))
       f3.write('\n')
    
    f3.close()
    
if __name__ == "__main__":
    main()

