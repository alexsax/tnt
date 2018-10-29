import torch
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from . import MeterLogger
from .. import meter as Meter
import numpy as np

class VisdomMeterLogger(MeterLogger):
    ''' A class to package and visualize meters.

    Args:
        server: The uri of the Visdom server
        env: Visdom environment to log to.
        port: Port of the visdom server.
        title: The title of the MeterLogger. This will be used as a prefix for all plots.
        plotstylecombined: Whether to plot train/test curves in the same window.
    '''
    def __init__(self, server="localhost", env='main', port=8097, title="DNN", nclass=21, plotstylecombined=True, log_to_filename=None):
        super(VisdomMeterLogger, self).__init__()        
        self.server = server
        self.port = port
        self.env = env
        self.title = title
        self.logger = {'Train': {}, 'Test': {}}
        self.plotstylecombined = plotstylecombined
        self.log_to_filename = log_to_filename

    def __addlogger(self, meter, ptype):
        if ptype == 'stacked_line':
            opts = {'title': self.title + ' ' + meter + ' (train)', 'fillarea': True, 'legend': self.meter[meter].keys}
            self.logger['Train'][meter] = VisdomPlotLogger(ptype, env=self.env, server=self.server,
                                                           port=self.port, log_to_filename=self.log_to_filename,
                                                           opts=opts)
            opts = {'title':self.title + ' ' + meter + ' (test)', 'fillarea': True, 'legend': self.meter[meter].keys}
            self.logger['Test'][meter] = VisdomPlotLogger(ptype, env=self.env, server=self.server,
                                                           port=self.port, log_to_filename=self.log_to_filename,
                                                           opts=opts)
        if ptype == 'line':
            if self.plotstylecombined:
                opts = {'title': self.title + ' ' + meter}
                self.logger['Train'][meter] = VisdomPlotLogger(ptype, env=self.env, server=self.server,
                                                               port=self.port, log_to_filename=self.log_to_filename,
                                                               opts=opts)
                opts = {}
                self.logger['Test'][meter] = self.logger['Train'][meter]
            else:
                opts = {'title': self.title + 'Train ' + meter}
                self.logger['Train'][meter] = VisdomPlotLogger(ptype, env=self.env, server=self.server,
                                                               port=self.port, log_to_filename=self.log_to_filename,
                                                               opts=opts)
                opts = {'title': self.title + 'Test ' + meter}
                self.logger['Test'][meter] = VisdomPlotLogger(ptype, env=self.env, server=self.server,
                                                               port=self.port, log_to_filename=self.log_to_filename,
                                                               opts=opts)
        elif ptype == 'heatmap':
            names = list(range(self.nclass))
            opts = {'title': self.title + ' Train ' + meter, 'columnnames': names, 'rownames': names}
            self.logger['Train'][meter] = VisdomLogger('heatmap', env=self.env, server=self.server,
                                                       port=self.port, log_to_filename=self.log_to_filename,
                                                       opts=opts)
            opts = {'title': self.title + ' Test ' + meter, 'columnnames': names, 'rownames': names}
            self.logger['Test'][meter] = VisdomLogger('heatmap', env=self.env, server=self.server,
                                                       port=self.port, log_to_filename=self.log_to_filename,
                                                       opts=opts)

    def add_meter(self, meter_name, meter):
        super(VisdomMeterLogger, self).add_meter(meter_name, meter)      
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
                self.logger[mode][key].log( np.array([iepoch]*len(val)), np.array(np.cumsum(val)), name=mode) # keep mean
            else:
                self.logger[mode][key].log(iepoch, val, name=mode)
            self.meter[key].reset()

