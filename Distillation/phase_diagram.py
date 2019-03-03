import matplotlib.pyplot as plt
import random

plt.xlim(xmax = 100, xmin = 0)
plt.ylim(ymax = 105, ymin = 75)
plt.rcParams['savefig.dpi'] = 2000
plt.rcParams['figure.dpi'] = 2000
plt.xlabel('x, y (w%)')
plt.ylabel('t (°C)')

author = [#'Evans',
'Carey', 'Noyes']

def clr(x, y):
    rgb = 0.2 + 0.8/len(author)*x
    return (rgb, 1-rgb, y) 


for au in range(len(author)):
    color1 = clr(au, 0.2)
    color2 = clr(au, 0.8)
    with open('phase_diagram_' + author[au] + '.txt') as f:
        data = f.readlines()
        for ln in range(len(data)):
            line = eval('[' + data[ln] + ']')
            if line[1] != None:
                plt.scatter(line[1], line[0], s=3, c=color1)
            if line[2] != None:
                plt.scatter(line[2], line[0], s=3, c=color2)

plt.savefig('./phase_diagram.png')
plt.show()