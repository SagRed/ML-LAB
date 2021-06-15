import matplotlib.pyplot as plt
import numpy as np


def estimate_coefficient(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)


def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="m", marker="o", s=30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./graph.png')


def main():
    x = np.array([43, 21, 25, 42, 57, 59])
    y = np.array([95, 65, 79, 75, 87, 81])
    b = estimate_coefficient(x, y)
    print("------------------------------")
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
    new_x = np.append(x, [55])
    unknown_y = (b[0] + 55*b[1])
    new_y = np.append(y, unknown_y)
    print("")
    print("x: 55,y:", unknown_y)

    print("------------------------------")
    print("")
    print("----graph as beed saved to the current folder---")
    print("")
    plot_regression_line(new_x, new_y, b)


if __name__ == "__main__":
    main()
