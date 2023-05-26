import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
import csv

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def main():

# Read the CSV file
    with open("data.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        T = []
        values = []
        for row in reader:
            T.append(row[0])
            values.append((row[1:]))

    # Convert the arrays to NumPy arrays
    T = np.array(T[1:])
    b = np.array(values[0],dtype=float)
    values = np.array(values[1:],dtype=float)

    # Print the arrays
    print(T)
    print(b)
    print(values)

    fig, ax = plt.subplots(figsize=[5,4])

    colors = ['#a50026','#d73027','#f46d43','#fdae61','#abd9e9','#74add1','#4575b4','#313695']

    for num_T,T_i in enumerate(T):
        label = r'$T$ = '+T_i
        plt.plot(b,values[num_T],label=label, color = colors[num_T])
    # plt.plot(b,values[0],label=r'$T$ = '+T[0],color='#d7191c')
    # plt.plot(b,values[1],label=r'$T$ = '+T[1],color='#ff7f00')
    # plt.plot(b,values[2],label=r'$T$ = '+T[2],color='#ffff33')
    # plt.plot(b,values[3],label=r'$T$ = '+T[3],color='#4daf4a')
    # plt.plot(b,values[4],label=r'$T$ = '+T[4],color='#2b83ba')
    # plt.plot(b,values[5],label=r'$T$ = '+T[5],color='#984ea3')
    # plt.plot(b,values[6],label=r'$T$ = '+T[6],color='#f781bf')
    # plt.plot(b,values[7],label=r'$T$ = '+T[7],color='#a65628')
    plt.xlabel(r'$b$')
    plt.ylabel(r'$\varepsilon$')
    plt.title(r'Ergodic Metric as Function of $b$ and $T$')
    plt.xlim([0,1])
    plt.ylim([0,0.19])
    plt.legend(fancybox=False, edgecolor='0')
    plt.tight_layout()

    axins = zoomed_inset_axes(ax, 3.2, loc="upper center") # zoom = 6

    for num_T,T_i in enumerate(T):
        axins.plot(b,values[num_T], color = colors[num_T])

    # axins.plot(b,values[0],label=r'$T$ = '+T[0],color='#d7191c')
    # axins.plot(b,values[1],label=r'$T$ = '+T[1],color='#ff7f00')
    # axins.plot(b,values[2],label=r'$T$ = '+T[2],color='#ffff33')
    # axins.plot(b,values[3],label=r'$T$ = '+T[3],color='#4daf4a')
    # axins.plot(b,values[4],label=r'$T$ = '+T[4],color='#2b83ba')
    # axins.plot(b,values[5],label=r'$T$ = '+T[5],color='#984ea3')
    # axins.plot(b,values[6],label=r'$T$ = '+T[6],color='#f781bf')
    # axins.plot(b,values[7],label=r'$T$ = '+T[7],color='#a65628')

    # sub region of the original image
    x1, x2, y1, y2 = 0, 0.125, 0, 0.04
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks(ticks=[0,0.025,0.05, 0.075, 0.1, 0.125], labels=[0,"", 0.05,"", 0.1,""])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.show()

    return

if __name__=='__main__':
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    exit(main())