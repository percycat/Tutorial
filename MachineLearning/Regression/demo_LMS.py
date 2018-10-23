"""
===========
Ordinary Least Square Demo
===========

Using the slider widget to control visual properties of your plot.

In this example, a slider is used to choose the variance of the distribution that
the noise belongs to.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
a0 = 1.5
f0 = -2.5
x_range = 5
plt.axis([0, x_range, -10, 10])
t = np.arange(0.0, x_range, 0.01)
s = a0*t + f0
l, = plt.plot(t, s, lw=2, color='red')
sample = np.random.normal(0, 1, len(t[0::10]))
N=plt.scatter(t[0::10], s[0::10]+sample, color='green')

axcolor = 'lightgoldenrodyellow'
ax_var = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

delta = 0.1
sVar = Slider(ax_var, 'variance', 0.1, 5.0, valinit=1.0, valstep=delta)

def draw(var, color='red'):
    ax.cla()
    ax.axis([0, x_range, -10, 10])
    ax.scatter(t[0::10], s[0::10]+sample, color='green')
    ax.plot(t, s, lw=2, color=color)
    fig.canvas.draw_idle()
    
def update(val):
    global sample
    var = sVar.val
    sample = np.random.normal(0, var, len(t[0::10]))
    draw(var)
    
sVar.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sVar.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'orange'), active=0)


def colorfunc(label):
    draw(sVar.val, label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()
