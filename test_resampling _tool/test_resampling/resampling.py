#-*- coding:utf-8 -*-#
# Copyright (c) National University of Defense Technology(NUDT).
# All rights reserved.
#
"""
Created on 2020-01-11

@author: zhangzhuo

usage : 
    python resampling.py dev
    or
    python resampling.py
"""
import sys
import numpy as np
import math
import random
from configparser import ConfigParser
import logging.config


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
        
        resample_coverage_information_matrix_section = sections[2]
        self.covMatrix_resample = config_parser.get(resample_coverage_information_matrix_section, "covMatrix_resample")
        
        resample_coverage_information_matrix_section = sections[3]
        self.error_resample = config_parser.get(resample_coverage_information_matrix_section, "error_resample")
        
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
    f3 = open(runtime_context.covMatrix_resample,'w')
    f4 = open(runtime_context.error_resample,'w')
    """ 
    load covMatrix.txt and error.txt
    """
    logger.info("load coverage information matrix")
    first_ele = True
    for data in f1.readlines():
        f3.write(str(data))
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [int(x) for x in nums]
            matrix_x = np.array(nums)
            first_ele = False
        else:
            nums = [int(x) for x in nums]
            matrix_x = np.c_[matrix_x,nums]
    f1.close()
    logger.info("test cases results.txt")
    first_ele = True
    for data in f2.readlines():
        f4.write(str(data))
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [int(x) for x in nums]
            matrix_y = np.array(nums)
            first_ele = False
        else:
            nums = [int(x) for x in nums]
            matrix_y = np.c_[matrix_y,nums] 
    f2.close()
    
    
    matrix = matrix_x.transpose()
    result = matrix_y.transpose()
    """ 
    compute total test cases' number, failing test cases' number and theta
    """
    logger.info("compute total test cases' number, failing test cases' number and theta")
    result_fail_list = []
    result_fail_number = 0
    result_number = len(result)
    
    for index in range(result_number):
        if result[index] == 1:
            result_fail_list.append(index)
            result_fail_number = result_fail_number +1
    
    theta = math.floor((result_number-2*result_fail_number)/result_fail_number)
    left_number = result_number-2*result_fail_number - theta*result_fail_number
    """ 
    generate new matrix file and error file
    """
    logger.info(" generate new matrix file and error file")
    for epoch in range(theta):
        for item in result_fail_list:
            for item_statement in matrix[item]:
                f3.write(str(item_statement))
                f3.write(' ')
            f3.write('\n')
            f4.write('1')
            f4.write('\n')
    
    if not left_number == 0:
        for epoch in range(left_number):
            random_choice_num = random.choice(result_fail_list)
            for item_statement in matrix[random_choice_num]:
                f3.write(str(item_statement))
                f3.write(' ')
            f3.write('\n')
            f4.write('1')
            f4.write('\n')
        
    f3.close()
    f4.close()
    logger.info(" generate complete")
if __name__ == "__main__":
    main()
