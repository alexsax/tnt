import os
import torch
import torchnet as tnt
from . import MeterLogger
from .. import meter as Meter
import numpy as np
import functools

IS_IMPORTED_TENSORBOARDX = False
try:
    import tensorboardX
    IS_IMPORTED_TENSORBOARDX = True
except:
    pass

class TensorboardMeterLogger(MeterLogger):
    ''' A class to package and visualize meters.

    Args:
        server: The uri of the Visdom server
        env: Visdom environment to log to.
        port: Port of the visdom server.
        title: The title of the MeterLogger. This will be used as a prefix for all plots.
        plotstylecombined: Whether to plot train/test curves in the same window.
    '''
    def __init__(self, env, log_dir=None, plotstylecombined=True):
        super().__init__() 
        self.logger = {'Train': {},
                       'Test': {}}
        self.env = env
        self.log_dir = os.path.join(log_dir, env)
        self.writer = {'Train': tensorboardX.SummaryWriter(log_dir=self.log_dir + "-train"),
                       'Test': tensorboardX.SummaryWriter(log_dir=self.log_dir + "-test")}
        self.plotstylecombined = plotstylecombined
        
    def __addlogger(self, meter, ptype):
        if ptype == 'stacked_line':
            raise NotImplementedError("stacked_line not yet implemented for TensorboardX meter")
        if ptype == 'line':
            if self.plotstylecombined:
                self.logger['Train'][meter] = functools.partial(self.writer['Train'].add_scalar, tag=meter)
                self.logger['Test'][meter] = functools.partial(self.writer['Test'].add_scalar, tag=meter)
            else:
                self.logger['Train'][meter] = functools.partial(self.writer['Train'].add_scalar, tag=meter + " train")
                self.logger['Test'][meter] = functools.partial(self.writer['Test'].add_scalar, tag=meter + " test")

        elif ptype == 'heatmap':
            raise NotImplementedError("heatmap not yet implemented for TensorboardX meter")


    def add_meter(self, meter_name, meter):
        super().add_meter(meter_name, meter)      
        if isinstance(meter, Meter.ClassErrorMeter):
            self.__addlogger(meter_name, 'line')
        elif isinstance(meter, Meter.mAPMeter):
            self.__addlogger(meter_name, 'line')
        elif isinstance(meter, Meter.AUCMeter):
            self.__addlogger(meter_name, 'line')
        elif isinstance(meter, Meter.ConfusionMeter):
            self.__addlogger(meter_name, 'heatmap')
        elif isinstance(meter, Meter.MSEMeter):
            self.__addlogger(meter_name, 'line')
        elif type(meter) == Meter.ValueSummaryMeter:
            self.__addlogger(meter_name, 'line')
        elif isinstance(meter, Meter.MultiValueSummaryMeter):
            self.__addlogger(meter_name, 'stacked_line')

    def reset_meter(self, iepoch, mode='Train'):
        self.timer.reset()
        for key, meter in self.meter.items():
            val = self.meter[key].value()
            val = val[0] if isinstance(val, (list, tuple)) else val
            if isinstance(meter, Meter.ConfusionMeter) or \
                key in ['histogram', 'image']:
                self.logger[mode][key].log(val)
            elif isinstance(self.meter[key], Meter.MultiValueSummaryMeter):
                self.logger[mode][key](scalar_value=np.array(np.cumsum(val), global_step=iepoch)) # keep mean
            else:
                self.logger[mode][key](scalar_value=val, global_step=iepoch)
            self.meter[key].reset()

