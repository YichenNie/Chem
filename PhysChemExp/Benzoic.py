import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize, interpolate, integrate


names = ['t/s', 'ΔT/°C']
df = pd.read_excel('Benzoic.xlsx', skiprows=0, names=names)
print(df)

# 前期和末期
startx = df['t/s'][:16]
starty = df['ΔT/°C'][:16]
endx = df['t/s'][51:]
endy = df['ΔT/°C'][51:]


# 线性拟合
def lin(x, A, B):
    return A * x + B
a1, b1 = optimize.curve_fit(lin, startx, starty)[0]
a2, b2 = optimize.curve_fit(lin, endx, endy)[0]

# 曲线拟合, 插值
# curve = interpolate.interp1d(df['t/s'], df['ΔT/°C'], kind='quadratic')    # interp1d两侧易产生震荡, Hermite插值更稳定
curve = interpolate.PchipInterpolator(df['t/s'], df['ΔT/°C'])


# 面积积分
def Area(DownLin, UpLin, FitCurve, Type, x, start, end):
    if Type == 'up':
        return integrate.quad(lambda y: UpLin(y)-FitCurve(y), x, end)[0]
    else:
        return integrate.quad(lambda y: FitCurve(y)-DownLin(y), start, x)[0]


#二分法查找时间
def FindTime(DownLin, UpLin, FitCurve, leftguess, rightguess, start, end):
    print(leftguess, rightguess)
    if rightguess - leftguess < 0.01:
        return rightguess
    else:
        mid = (leftguess + rightguess) / 2
        if Area(DownLin, UpLin, FitCurve, 'up', mid, start, end) >= Area(DownLin, UpLin, FitCurve, 'down', mid, start, end):
            return FindTime(DownLin, UpLin, FitCurve, mid, rightguess, start, end)
        else:
            return FindTime(DownLin, UpLin, FitCurve, leftguess, mid, start, end)

Time = FindTime(lambda x: lin(x, a1, b1), lambda x: lin(x, a2, b2), curve, df['t/s'][16], df['t/s'][50], df['t/s'][15], df['t/s'][51])


# 准备作图
plt.figure(figsize=(6, 4.5), dpi=320)
plt.axis([0, 1000, -0.1, 2.0])

# 数据-时间曲线
plt.plot(startx, lin(startx, a1, b1), linewidth=0.5, color=[0.7,0.2,0.2])
plt.plot(endx, lin(endx, a2, b2), linewidth=0.5, color=[0.7,0.2,0.2])

left, right = df['t/s'][15], df['t/s'][51]
plt.plot(np.linspace(left, right, 500), curve(np.linspace(left, right, 500)), linewidth=1, color=[0.7,0.2,0.2])

# 辅助线
plt.plot(np.linspace(200, 400, 10), lin(np.linspace(200, 400, 10), a1, b1), linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
plt.plot(np.linspace(300, 800, 10), lin(np.linspace(300, 800, 10), a2, b2), linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
plt.vlines(Time, ymin=-0.1, ymax=lin(Time, a2, b2), linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
plt.vlines(df['t/s'][15], ymin=-0.1, ymax=df['ΔT/°C'][16], linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
plt.vlines(df['t/s'][51], ymin=-0.1, ymax=df['ΔT/°C'][51], linewidth=1, color=[0.7,0.2,0.2], linestyle='--')

# 数据点
plt.scatter(df['t/s'], df['ΔT/°C'], c=[0.3,0.3,0.3], s=4, marker=',')

# 反应温度差
DeltaT = lin(Time, a2, b2) - lin(Time, a1, b1)
print(DeltaT, Time)

# 坐标轴粗细
ax = plt.subplot(1,1,1)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)

# 字体
font1 = {'family' : 'Arial',
'weight' : 'normal',
'size' : 10,
}
plt.tick_params(labelsize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]

plt.xlabel('$t$/s', font1)
plt.ylabel('Δ$T$/°C', font1)
plt.grid(True, linewidth=0.5)
plt.savefig("Benzoic.png", transparent=True)
