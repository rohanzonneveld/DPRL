import numpy as np

X = [i/100 for i in range(100, 9, -1)]
equation1 = np.array([[1] + [0] * 89 + [1]])
coeffs = np.array(equation1)
consts = np.ones(91)

for i in range(1,len(X)):
    equation = np.zeros(91)
    equation[i] = 1
    equation[-1] = coeffs[-1][-1] * (1-X[i]) + 1
    coeffs = np.vstack((coeffs, equation))
    if i == 89:
        print(equation)

solution = np.linalg.solve(coeffs, consts)
print(solution)


