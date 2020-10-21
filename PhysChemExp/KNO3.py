import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize, interpolate, integrate


# 时间节点
timedata = [
    [10, 51, 60, 96],
    [10, 31, 41, 75],
    [10, 25, 35, 73],
    [11, 24, 34, 53],
    [11, 23, 35, 54],
    [11, 26, 36, 61],
]

for i in range(1, 7):
    sheetname = str(i)

    names = ['t/s', 'ΔT/°C']
    df = pd.read_excel('KNO3.xlsx', skiprows=0, names=names, sheet_name=sheetname)


    # 加样时间
    # 温度补偿前平台期开始
    # 温度补偿前平台期结束
    # 温度补偿后平台期开始
    addtime, midtimestart, midtimeend, endtime = timedata[i-1]
    length=len(df)
    print(length)
    print(df['ΔT/°C'][addtime], df['ΔT/°C'][midtimestart], df['ΔT/°C'][midtimeend], df['ΔT/°C'][endtime])

    startx = df['t/s'][:addtime]
    starty = df['ΔT/°C'][:addtime]
    midx = df['t/s'][midtimestart:midtimeend+1]
    midy = df['ΔT/°C'][midtimestart:midtimeend+1]
    endx = df['t/s'][endtime:]
    endy = df['ΔT/°C'][endtime:]


    # 线性拟合
    def lin(x, A, B):
        return A * x + B
    a1, b1 = optimize.curve_fit(lin, startx, starty)[0]
    a2, b2 = optimize.curve_fit(lin, midx, midy)[0]
    a3, b3 = optimize.curve_fit(lin, endx, endy)[0]

    # 曲线拟合, 插值
    # curve = interpolate.interp1d(df['t/s'], df['ΔT/°C'], kind='quadratic')    # interp1d两侧易产生震荡, Hermite插值更稳定
    curve = interpolate.PchipInterpolator(df['t/s'], df['ΔT/°C'])


    # 面积积分
    def Area(Lin, FitCurve, Type, start, end):
        if Type == 'up':
            return integrate.quad(lambda y: Lin(y)-FitCurve(y), start, end)[0]
        else:
            return integrate.quad(lambda y: FitCurve(y)-Lin(y), start, end)[0]


    #二分法查找时间
    def FindTime(DownLin, UpLin, FitCurve, leftguess, rightguess, start, end, typ):
        if abs(rightguess - leftguess) < 0.01:
            return rightguess
        else:
            
            mid = (leftguess + rightguess) / 2
            if typ == 'heat':
                if Area(UpLin, FitCurve, 'up', mid, end) >= Area(DownLin, FitCurve, 'down', start, mid):
                    return FindTime(DownLin, UpLin, FitCurve, mid, rightguess, start, end, typ)
                else:
                    return FindTime(DownLin, UpLin, FitCurve, leftguess, mid, start, end, typ) 
            else:
                if Area(UpLin, FitCurve, 'up', start, mid) >= Area(DownLin, FitCurve, 'down', mid, end):
                    return FindTime(DownLin, UpLin, FitCurve, leftguess, mid, start, end, typ)
                else:
                    return FindTime(DownLin, UpLin, FitCurve, mid, rightguess, start, end, typ)

    Time1 = FindTime(lambda x: lin(x, a2, b2), lambda x: lin(x, a1, b1), curve, df['t/s'][addtime+1], df['t/s'][midtimestart-1], df['t/s'][addtime], df['t/s'][midtimestart], 'dissolve')
    Time2 = FindTime(lambda x: lin(x, a2, b2), lambda x: lin(x, a3, b3), curve, df['t/s'][midtimeend+1], df['t/s'][endtime-1], df['t/s'][midtimeend], df['t/s'][endtime], 'heat')


    # 准备作图
    plt.figure(figsize=(6, 4.5), dpi=320)
    axisparam=[0, max(df['t/s']), math.floor(min(df['ΔT/°C']*5-1))/5, math.ceil(max(df['ΔT/°C']*5+1))/5]
    plt.axis(axisparam)

    # 数据-时间曲线
    plt.plot(np.linspace(df['t/s'][0], df['t/s'][length-1], 3000), curve(np.linspace(df['t/s'][0], df['t/s'][length-1], 3000)), linewidth=1, color=[0.7,0.2,0.2])

    # 辅助线
    plt.plot(np.linspace(df['t/s'][0], Time1, 10), lin(np.linspace(df['t/s'][0], Time1, 10), a1, b1), linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
    plt.plot(np.linspace(Time1, Time2, 10), lin(np.linspace(Time1, Time2, 10), a2, b2), linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
    plt.plot(np.linspace(Time2, df['t/s'][length-1], 10), lin(np.linspace(Time2, df['t/s'][length-1], 10), a3, b3), linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
    plt.vlines(Time1, ymin=lin(Time1, a2, b2), ymax=lin(Time1, a1, b1), linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
    plt.vlines(Time2, ymin=lin(Time2, a2, b2), ymax=lin(Time2, a3, b3), linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
    # plt.vlines(df['t/s'][addtime], ymin=axisparam[2], ymax=df['ΔT/°C'][addtime], linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
    # plt.vlines(df['t/s'][midtimestart], ymin=axisparam[2], ymax=df['ΔT/°C'][midtimestart], linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
    # plt.vlines(df['t/s'][midtimeend], ymin=axisparam[2], ymax=df['ΔT/°C'][midtimeend], linewidth=1, color=[0.7,0.2,0.2], linestyle='--')
    # plt.vlines(df['t/s'][endtime], ymin=axisparam[2], ymax=df['ΔT/°C'][endtime], linewidth=1, color=[0.7,0.2,0.2], linestyle='--')

    # 数据点
    plt.scatter(df['t/s'], df['ΔT/°C'], c=[0.3,0.3,0.3], s=4, marker=',')

    # 反应温度差
    DeltaT1 = lin(Time1, a1, b1) - lin(Time1, a2, b2)
    print(lin(Time1, a1, b1), lin(Time1, a2, b2), DeltaT1, Time1)
    DeltaT2 = lin(Time2, a3, b3) - lin(Time2, a2, b2)
    print(lin(Time2, a3, b3), lin(Time2, a2, b2), DeltaT2, Time2)


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

    # 坐标轴
    yax = np.linspace(axisparam[2], axisparam[3], round((axisparam[3]-axisparam[2])/0.2+1))
    yaxlabels = ['%.3f' % i for i in yax]
    plt.yticks(yax, yaxlabels)

    plt.xlabel('$t$/s', font1)
    plt.ylabel('Δ$T$/°C', font1)
    plt.grid(True, linewidth=0.5)
    plt.savefig(sheetname + ".png", transparent=True)