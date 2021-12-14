from with_numpy import run_main as r1
from without_numpy import run_main as r2
import numpy as np

def compare_results(res1,res2):
    a,b,c,d,e=res1
    a1,b1,c1,d1,e1=res2
    c = np.around(c, 0)
    c1 = np.around(c1, 0)
    d = np.around(d, 12)
    d1 = np.around(d1, 12)
    e = np.around(e, 12)
    e1 = np.around(e1, 12)
    return (a==a1 and b==b1 and np.array_equal(c,c1) and np.array_equal(d,d1) and np.array_equal(e,e1))

numpy_result=r1()
python_result=r2()
print("Numpy:",numpy_result)
print("Python:",python_result)
print("Are the following fields equal [determinant,matrix_transpose,matrix_adjugate,matrix_inverse,final_result]:",compare_results(numpy_result,python_result))