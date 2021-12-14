import re
from enum import IntEnum
import numpy as np


class c_pos(IntEnum):
    x = 0
    y = 1
    z = 2


coeficients_matrix = np.empty(shape=(3, 3), dtype=int)
equations_results = np.empty(3, dtype=int)


def process_file():
    counter = 0
    with open("Sistem.txt") as f:
        for line in f:
            tokens = re.split("([xyz=])", line.strip())
            result = tokens[len(tokens) - 1]
            for index, element in enumerate(tokens):
                if index + 1 < len(tokens) and index - 1 >= 0:
                    if element == 'x' or element == 'y' or element == 'z':
                        prev_el = str(tokens[index - 1])
                        if prev_el == '+' or prev_el == '':
                            coeficients_matrix[counter][c_pos[element]] = 1
                        elif prev_el == '-':
                            coeficients_matrix[counter][c_pos[element]] = -1
                        else:
                            coeficients_matrix[counter][c_pos[element]] = int(prev_el)
            print("Line tokens:", tokens)
            print(result)
            equations_results[counter] = result
            counter = counter + 1
        print("Coeficient matrix:\n", coeficients_matrix)
        print("Equation results:", equations_results)


def compute_determinant():
    global determinant
    determinant = round(np.linalg.det(coeficients_matrix))
    if determinant == 0:
        print("Determinant is 0!")
        exit()
    else:
        return determinant


def compute_transpose():
    global matrix_transpose
    matrix_transpose = coeficients_matrix.transpose()
    return matrix_transpose


def compute_adjugate():
    global matrix_adjugate,matrix_inverse
    matrix_inverse=compute_inverse()
    matrix_adjugate = matrix_inverse * determinant
    return matrix_adjugate



def compute_inverse():
    global matrix_inverse
    matrix_inverse = np.linalg.inv(coeficients_matrix)
    return matrix_inverse


def run_main():
    process_file()
    compute_determinant()  # Not required
    compute_transpose()    # Not required
    compute_adjugate()     # Not required
    compute_inverse()
    final_result = matrix_inverse.dot(equations_results)
    # print("[1][x,y,z]:", final_result)
    # final_result_2 = np.linalg.solve(coeficients_matrix, equations_results)
    # print("[2][x,y,z]:", final_result_2)
    return determinant,matrix_transpose.tolist(),matrix_adjugate.tolist(),matrix_inverse.tolist(),final_result.tolist()


# ########### With numpy start#############
# process_file()
# print("Determinant of the coeficient matrix:",compute_determinant()) #Not required
# print("Transpose:\n",compute_transpose())                            #Not required
# print("Adjugate:\n", compute_adjugate())                             #Not required
# print("Inverse:\n", compute_inverse())
#
# final_result = matrix_inverse.dot(equations_results)
# print("[x,y,z]:", final_result)
# final_result_2 = np.linalg.solve(coeficients_matrix, equations_results)
# print("[2][x,y,z]:", final_result_2)
# ########### With numpy end#############
