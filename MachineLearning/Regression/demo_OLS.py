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
t = np.arange(0.0, 10.0, 0.01)
a0 = 1.5
f0 = -7.5

s = a0*t + f0
l, = plt.plot(t, s, lw=2, color='red')
plt.axis([0, 10, -10, 10])

axcolor = 'lightgoldenrodyellow'
ax_var = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

delta = 0.1
sVar = Slider(ax_var, 'variance', 0.1, 5.0, valinit=1.0, valstep=delta)


def update(val):
    var = sVar.val
    fig.canvas.draw_idle()
sVar.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sVar.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()
