import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import os
import math
import h5py
import numpy as np
import train_utilities
from decimal import *

origin_mic = 0
n_reflections = 'First'
Room = '12BB01'
set_precision = 4
angle_step = 10
total_positions = 360/angle_step

def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 1.
    else:
        return x

def roundup(x, base=angle_step):
    a = int(base * round(float(x)/base))/base
    if a == total_positions:
       a = 0
    return a

if Room == 'A':
    l=5.720
    w=6.640
    r=1.500
    a=1.500
    b=2.860

if Room == 'D':
    l=8.020
    w=8.720
    r=1.500
    a=3.730
    b=4.360

if Room == '12BB01':
    l=5.51
    w=4.23
    r=1.500
    a=2.75
    b=2.12

x=l-a
y=w-b


def draw_room():
    
    head = [0,0]
    image_points = []
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, aspect='equal')
    
    plt.tick_params(labelsize=18)
    plt.title('Room setup', fontsize=30)
    plt.xlabel('Length (m)', fontsize=20)
    plt.ylabel('Width (m)', fontsize=20)
    
    if origin_mic == 1:
        
       ax.add_patch( patches.Rectangle((x-l, y-w), l, w, fill=True) )
       plt.plot(head[0], head[1], 'o', c='black')
    
    elif origin_mic == 0:
       ax.add_patch( patches.Rectangle((x-l+a, y-w+b), l, w, fill=True, facecolor = '#85bab2', edgecolor='black',linewidth = 3) )
       plt.plot(head[0]+a, head[1]+b, 'x', c='black')
    
    angles = range(-180,180,10)
    colors=["#0000FF", "#00FF00", "#FF0066"]
    i=0
    
    for theta in angles:
        print(theta)
        
        rx = r*np.cos(np.deg2rad(theta) )
        ry = r*np.sin(np.deg2rad(theta) )
        s = [round(rx,set_precision), round(ry,set_precision)]
        
        if origin_mic == 0:
            
           s[0] = s[0]+a
           s[1] = s[1]+b
        
        print(s)

        plt.plot(s[0], s[1],'X', color='white', markersize=12) 
        
        i+=1
        

    if origin_mic == 0:
       savefig_name = os.path.join( os.getcwd(),  'Room' + Room + '_origininroom' )
       
    elif origin_mic == 1:
       savefig_name = os.path.join( os.getcwd(),  'RoomA' + Room + '_origininmicrophones' )
   
    plt.savefig(savefig_name)
    plt.clf()
    plt.show()


if __name__ == "__main__":
    
    draw_room()
