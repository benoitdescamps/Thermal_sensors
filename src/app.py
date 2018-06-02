'''
Bokeh app for streaming result

--> to do
--> compare with tensorboard
'''

import numpy as np
np.random.seed(1)

from bokeh.models import ColumnDataSource, Slider, Select
from bokeh.plotting import curdoc, figure
from bokeh.driving import count

BUFSIZE = 200

source = ColumnDataSource(dict(
    time=[], image=[]
))

p = figure(plot_height=500)


@count()
def update(t):
    new_data = dict(
        time=[t],
        image=[0])

    source.stream(new_data, 300)

curdoc().add_periodic_callback(update, 50)
curdoc().title = "Thermal Sensor"