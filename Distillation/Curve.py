import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import optimize


with open('distillation.txt', 'r') as d:
    data = d.readlines()
    v = []
    t = []
    for i in range(len(data)):
        ln = eval('[' + data[i] + ']')
        v.append(ln[0])
        t.append(ln[1])
        plt.scatter(ln[0], ln[1], s=5, c=(0.8, 0.3, 0.3))
    x = np.array(v)
    y = np.array(t)
    curve = np.poly1d(np.polyfit(x, y, 13))
    xx = np.linspace(1, 24, 100)
    yy = curve(xx)
    plt.plot(xx, yy, c=(0.8, 0.3, 0.3), linestyle='solid')

with open('fractional_distillation.txt', 'r') as d:
    data = d.readlines()
    v = []
    t = []
    for i in range(len(data)):
        ln = eval('[' + data[i] + ']')
        v.append(ln[0])
        t.append(ln[1])
        plt.scatter(ln[0], ln[1], s=16, c=(0.3, 0.3, 0.8), marker='+')
    x = np.array(v)
    y = np.array(t)
    #curve = np.poly1d(np.polyfit(x, y, 10))
    xx = np.arange(1, 20, 0.2)
    '''yy = curve(xx)
    plt.plot(xx, yy, c=(0.3, 0.3, 0.8))'''
    plt.plot(x, y, c=(0.3, 0.3, 0.8), linestyle='dashed')


plt.title('Normal distillation (solid)\n Fractional distillation (dashed)')
plt.xlim(xmax = 25, xmin = 0)
plt.ylim(ymax = 100, ymin = 70)
plt.rcParams['savefig.dpi'] = 2000
plt.rcParams['figure.dpi'] = 2000
plt.xlabel('V (mL)')
plt.ylabel('t (°C)')
plt.savefig('curve.png')
plt.show()
