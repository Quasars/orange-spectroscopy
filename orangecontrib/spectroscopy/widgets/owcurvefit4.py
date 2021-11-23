from Orange.widgets.visualize.utils.plotutils import InteractiveViewBox
import numpy as np
from Orange.widgets.widget import OWWidget, Msg
import Orange.data
import Orange.data.table
from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler, SettingProvider
from orangecontrib.spectroscopy.widgets.owspectra import SELECTMANY, CurvePlot, InteractiveViewBoxC
from Orange.widgets.widget import OWWidget, Input
from AnyQt.QtCore import pyqtSignal
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from AnyQt.QtWidgets import QFormLayout, QWidget, QLayout


from PyQt5.QtWidgets import *
import sys
import pyqtgraph as pg
from PyQt5.QtGui import *

# TODO  
#       


class OWCurveFit(OWWidget):
    name = 'Curve Fit Plot'

    want_control_area = False

    #pg.mkBrush(255(r), 0(b), 0(g), 120(a))
    color_pallette =['#0000FF', #Blue
                    '#7FFF00',  #Chart
                    '#00FFFF',  #Cyan
                    '#006400',  #Dark Green
                    '#FF8C00',  #Dark Orange
                    '#000000',  #Black
                    '#9400D3',  #Dark Violet
                    '#FF0000',  #Red
                    '#A0522D',  #Sienna/Brown
                    '#FFFF00']  #Yellow


    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Warning(OWWidget.Warning):
        incompatible = Msg('Incoming data is not compatible with this widget')
    
    class Information(OWWidget.Information):
        show = Msg('Showing {} of {} curves')

    def __init__(self):
        super().__init__()

        # creating a widget object
        self.widget = QWidget()
  
        # creating a plot window
        self.plot = pg.plot()

        # creating a scatter plot item
        # of size = 10
        # using brush to enlarge the of white color with transparency is 50%
        

        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.line = pg.PlotCurveItem(pen=pg.mkPen(width=2))
        #test2.addPoints(pos=xy)
        #xy = np.empty((100,2))
        #xy[:,0]=np.linspace(0,1,100)
        #xy[:,1]=np.linspace(0,1,100)
        #tempdata = np.empty((1,2))
        #self.scatter.setData(pos=tempdata)
        # adding scatter plot item to the plot window
        #plot.addItem(test1)
        #self.plot.addItem(self.scatter)
        #self.plot.addItem(self.line)

        # Creating a grid layout
        layout = QGridLayout()
        
        # setting this layout to the widget
        self.widget.setLayout(layout)
  
        # plot window goes on right side, spanning 3 rows
        layout.addWidget(self.plot, 0, 1, 3, 1)
  
        # setting this widget as central widget of the main widow
        self.layout().addWidget(self.plot)
        

    @Inputs.data
    def set_data(self, data):
        self.Warning.incompatible.clear()

        self.data = data

        if data is None:
            return

        if 'a0' and 'a1' and 'a2' and '0 degrees' and '45 degrees' and '90 degrees' and '135 degrees' in data.domain:
            self.commit()
        else:
            self.Warning.incompatible()


    def Azimuth(self,x,a0,a1,a2):
        return a0*np.sin(2*np.radians(x))+a1*np.cos(2*np.radians(x))+a2

    def commit(self):

        self.Information.show.clear()

        if self.data is None:
            return

        self.plot.plotItem.clear()

        scattery = np.asarray([self.data[:,'0 degrees'].X.reshape(-1), self.data[:,'45 degrees'].X.reshape(-1), self.data[:,'90 degrees'].X.reshape(-1), self.data[:,'135 degrees'].X.reshape(-1)])
        scattery = np.transpose(scattery)
        scatterx = np.array([0,45,90,135])

        

        scat = []
        count = 0
        cp = 0
        for i in scattery:
            scat.append(pg.ScatterPlotItem(size=10, brush=pg.mkBrush(self.color_pallette[cp])))
            scat[count].setData(x=scatterx, y=i)
            self.plot.addItem(scat[count])
            count += 1
            cp += 1
            if cp > 9:
                cp = 0
            if count > 10:
                break
        
        linx = np.linspace(0,180,360)
        liny = np.empty((np.shape(self.data)[0],360))

        for i in range(len(liny[:,0])): #number of samples/rows
            for j in range(len(linx)): #x-values
                liny[i,j] = self.Azimuth(linx[j], self.data[i,'a0'].value, self.data[i,'a1'].value, self.data[i,'a2'].value)

        line = []
        count = 0
        cp = 0
        for i in liny:
            line.append(pg.PlotCurveItem(pen=pg.mkPen(self.color_pallette[cp], width=2)))
            line[count].setData(x=linx, y=i)
            self.plot.addItem(line[count])
            count += 1
            cp += 1
            if cp > 9:
                cp = 0
            if count > 10:
                self.Information.show(count-1, self.data.X.shape[0])
                break


        
