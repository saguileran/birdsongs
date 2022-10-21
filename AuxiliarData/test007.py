'''
import matplotlib.pyplot as plt
import numpy as np

def get_coords_from_figure():
    ev = None
    def onclick(event):
        nonlocal ev
        ev = event

    fig, ax = plt.subplots()
    ax.axvline(x=0.5)      # Placeholder data
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show(block=True)
    return (ev.xdata, ev.ydata) if ev is not None else None
    # return (ev.x, ev.y) if ev is not None else None
    
x=np.arange(10)
plt.plot(x)
get_coords_from_figure()
'''


%matplotlib widget
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

# set up plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_ylim([-4, 4])
ax.grid(True)
 
# generate x values
x = np.linspace(0, 2 * np.pi, 100)
 
 
def my_sine(x, w, amp, phi):
    """
    Return a sine for x with angular frequeny w and amplitude amp.
    """
    return amp*np.sin(w * (x-phi))
 
 
@widgets.interact(w=(0, 10, 1), amp=(0, 4, .1), phi=(0, 2*np.pi+0.01, 0.01))
def update(w = 1.0, amp=1, phi=0):
    """Remove old lines from plot and plot new one"""
    [l.remove() for l in ax.lines]
    ax.plot(x, my_sine(x, w, amp, phi), color='C0')