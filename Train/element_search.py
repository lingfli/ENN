# coding : utf-8
import numpy as np
# ����ڵ㣬���ذ�����һ�ڵ�ĵ�Ԫ
# nodeΪ��ڵ�
def ele_search(node, data_element):
    ele_list = []
    for ele in data_element:
        mark = 0
        if ele[1] == node:
            mark = 1
        if ele[2] == node:
            mark = 1
        if ele[3] == node:
            mark = 1
        if mark == 1:
            ele_list.append(int(ele[0]))
    ele_out = np.array(ele_list)
    return ele_out