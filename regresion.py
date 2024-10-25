import matplotlib.pyplot as plt
import numpy as np
X = np.array([2, 3, 5])
y = np.array([2, 5, 4])

def f(x, w, b):
    return w * x + b
def costo(w, b, x_vals, y_vals):
    m = len(x_vals)
    total_error = 0
    for i in range(m):
        prediccion = f(x_vals[i], w, b)
        error = (prediccion - y_vals[i]) ** 2
        total_error += error
    return total_error / (2 * m)
def derivada_costo(x_vals, y_vals, w, b):
    m = len(x_vals)
    dw = (1 / m) * np.sum(2*((f(x_vals, w, b) - y_vals) * x_vals))
    db = (1 / m) * np.sum(2*(f(x_vals, w, b) - y_vals))
    return dw, db

plt.scatter(X,y,c="#4ad66d")
plt.xlim((0,6))
plt.xlim((0,6))

w = 0.4
inicial = 0
aprendizaje = 0.1
interacciones = 1000

for iteraction in range(interacciones):
    dw, db = derivada_costo(X, y, w, inicial)
    inicial=inicial-(aprendizaje*db)

    y_predic=[]
    for x in X:
        y_predic.append(inicial+(0.4*x))
    plt.plot(X,y_predic,color="#00a8e8")

    error=[]
    for i in range(len(X)):
        error.append((y[i]-(inicial+(0.4*X[i])))**2)
    mse= (1/3)*np.sum(error)
    print(f"MSE: {mse} and inicial: {inicial}")
plt.show()  
