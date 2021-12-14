import re
from enum import IntEnum


class c_pos(IntEnum):
    x = 0
    y = 1
    z = 2


variables = ['x', 'y', 'z']
coeficients_matrix = []
equations_results = []
matrix_transpose=[]
matrix_inverse=[]

def process_file():
    with open("Sistem.txt") as f:
        for line in f:
            tokens = re.split("([xyz=])", line.strip())
            line_coeficients = [0, 0, 0]
            result = tokens[len(tokens) - 1]
            for index, element in enumerate(tokens):
                if index + 1 < len(tokens) and index - 1 >= 0:
                    if element == 'x' or element == 'y' or element == 'z':
                        prev_el = str(tokens[index - 1])
                        if prev_el == '+' or prev_el == '':
                            line_coeficients[c_pos[element]] = 1
                        elif prev_el == '-':
                            line_coeficients[c_pos[element]] = -1
                        else:
                            line_coeficients[c_pos[element]] = int(prev_el)
            print("Line tokens:", tokens)
            print("Coeficients:", line_coeficients)
            print(result)
            coeficients_matrix.append(line_coeficients)
            equations_results.append(int(result))
        print("Coeficient matrix:", coeficients_matrix)
        print(equations_results)


def compute_determinant():
    global determinant
    determinant = coeficients_matrix[0][0] * coeficients_matrix[1][1] * coeficients_matrix[2][2] + \
                  coeficients_matrix[1][0] * coeficients_matrix[2][1] * coeficients_matrix[0][2] + \
                  coeficients_matrix[0][1] * coeficients_matrix[1][2] * coeficients_matrix[2][0] - \
                  coeficients_matrix[0][2] * coeficients_matrix[1][1] * coeficients_matrix[2][0] - \
                  coeficients_matrix[0][1] * coeficients_matrix[1][0] * coeficients_matrix[2][2] - \
                  coeficients_matrix[0][0] * coeficients_matrix[2][1] * coeficients_matrix[1][2]
    if determinant == 0:
        print("Determinant is 0!")
        exit()
    else:
        return determinant


def compute_transpose():
    for j in range(columns):
        row = []
        for i in range(rows):
            row.append(coeficients_matrix[i][j])
        matrix_transpose.append(row)
    return matrix_transpose


def compute_adjugate():
    global matrix_adjugate
    matrix_adjugate = [
        [
            matrix_transpose[1][1] * matrix_transpose[2][2] - matrix_transpose[1][2] * matrix_transpose[2][1],
            -(matrix_transpose[1][0] * matrix_transpose[2][2] - matrix_transpose[1][2] * matrix_transpose[2][0]),
            matrix_transpose[1][0] * matrix_transpose[2][1] - matrix_transpose[1][1] * matrix_transpose[2][0]
        ], [
            -(matrix_transpose[0][1] * matrix_transpose[2][2] - matrix_transpose[0][2] * matrix_transpose[2][1]),
            matrix_transpose[0][0] * matrix_transpose[2][2] - matrix_transpose[0][2] * matrix_transpose[2][0],
            -(matrix_transpose[0][0] * matrix_transpose[2][1] - matrix_transpose[0][1] * matrix_transpose[2][0])
        ],
        [
            matrix_transpose[0][1] * matrix_transpose[1][2] - matrix_transpose[0][2] * matrix_transpose[1][1],
            -(matrix_transpose[0][0] * matrix_transpose[1][2] - matrix_transpose[0][2] * matrix_transpose[1][0]),
            matrix_transpose[0][0] * matrix_transpose[1][1] - matrix_transpose[0][1] * matrix_transpose[1][0]
        ]
    ]
    return matrix_adjugate


def compute_inverse():
    print(matrix_adjugate)
    for i in range(rows):
        row = []
        for j in range(columns):
            row.append(matrix_adjugate[i][j] / determinant)
        matrix_inverse.append(row)

    return matrix_inverse


def multiply_matrix_by_vector(m, v):
    result = []
    for i in range(len(m)):
        total = 0
        for j in range(len(v)):
            total += v[j] * m[i][j]
        result.append(total)
    return result


def run_main():
    process_file()
    compute_determinant()
    global rows,columns
    rows = len(coeficients_matrix)
    columns = len(coeficients_matrix[0])
    compute_transpose()
    compute_adjugate()
    compute_inverse()
    final_result = multiply_matrix_by_vector(matrix_inverse, equations_results)
    print("[x,y,z]:", final_result)
    return determinant,matrix_transpose,matrix_adjugate,matrix_inverse,list(final_result)


# ########### Without numpy start#############
# process_file()
# print("Determinant of the coeficient matrix:",compute_determinant())
# rows = len(coeficients_matrix)
# columns = len(coeficients_matrix[0])
# print("Transpose:\n",compute_transpose())
# print("Adjugate:\n", compute_adjugate())
# print("Inverse:\n", compute_inverse())
# final_result = multiply_matrix_by_vector(matrix_inverse, equations_results)
# print("[x,y,z]:", final_result)
# ########### Without numpy end#############
