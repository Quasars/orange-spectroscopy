import warnings
from scipy.optimize import OptimizeWarning

import numpy as np
from scipy.optimize import curve_fit, minpack
import math

import Orange.data
from Orange.data import DiscreteVariable, ContinuousVariable
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings
from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler, SettingProvider
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatOrNone
from Orange.widgets.utils.itemmodels import DomainModel, VariableListModel
from Orange.data.util import get_indices

from AnyQt.QtWidgets import QFormLayout, QWidget

from orangecontrib.spectroscopy.data import _spectra_from_image, getx, build_spec_table
from orangecontrib.spectroscopy.utils import get_hypercube




def get_hypercubes(images, xy):
    output = []
    lsx, lsy = None, None
    for im in images:
        hypercube, lsx, lsy = get_hypercube(im, im.domain[xy[0]], im.domain[xy[1]])
        output.append(hypercube)
    return output, lsx, lsy
#Calculate by fitting to function
def Azimuth(x,a0,a1,a2):
    return a0*np.sin(2*np.radians(x))+a1*np.cos(2*np.radians(x))+a2

def calc_angles(a0,a1):
    return np.degrees(0.5*np.arctan(a0/a1))

def ampl1(a0,a1,a2):
    return (a2+(math.sqrt(a0**2+a1**2))+a2-(math.sqrt(a0**2+a1**2)))

def ampl2(a0,a1):
    return (2*(math.sqrt(a0**2+a1**2)))

def OrFunc(alpha,a0,a1,a2):
    if alpha == 0:
        Dmax = (2*a2+2*math.sqrt(a0**2+a1**2))/(2*a2-2*math.sqrt(a0**2+a1**2))
        return ((Dmax-1)/(Dmax+2)*(2/(3*np.cos(np.radians(alpha))**2-1)))
    elif alpha == 90:
        Dmin = (2*a2-2*math.sqrt(a0**2+a1**2))/(2*a2+2*math.sqrt(a0**2+a1**2))
        return ((Dmin-1)/(Dmin+2)*(2/(3*np.cos(np.radians(alpha))**2-1)))

def compute(images, wnidx, alpha):
    x = np.asarray([0,45,90,135])
    Az = np.empty(images[0].shape[:2])
    Az.fill(np.nan)
    Hermans = np.empty(images[0].shape[:2])
    Hermans.fill(np.nan)
    ampabs = np.empty(images[0].shape[:2])
    ampabs.fill(np.nan)
    funcamp = np.empty(images[0].shape[:2])
    funcamp.fill(np.nan)
    r_squared = np.empty(images[0].shape[:2])
    r_squared.fill(np.nan)
    a0 = np.empty((images[0].shape[:2]))
    a0.fill(np.nan)
    a1 = np.empty((images[0].shape[:2]))
    a1.fill(np.nan)
    a2 = np.empty((images[0].shape[:2]))
    a2.fill(np.nan)
    deg0 = np.empty((images[0].shape[:2]))
    deg0.fill(np.nan)
    deg45 = np.empty((images[0].shape[:2]))
    deg45.fill(np.nan)
    deg90 = np.empty((images[0].shape[:2]))
    deg90.fill(np.nan)
    deg135 = np.empty((images[0].shape[:2]))
    deg135.fill(np.nan)


    for i in range(images[0].shape[0]):
        for j in range(images[0].shape[1]):
            if np.isnan(images[0][i,j,wnidx]) == True:
                continue
            temp = [images[0][i,j,wnidx],images[1][i,j,wnidx],images[2][i,j,wnidx],images[3][i,j,wnidx]]
            temp = np.reshape(np.asarray(temp,dtype=float),-1)

            params, cov = curve_fit(Azimuth, x, temp)

            residuals = temp - Azimuth(x, *params)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((temp-np.mean(temp))**2)
            r_squared[i,j] = 1-(ss_res/ss_tot)

            Az0 = calc_angles(params[0],params[1])
            Abs0 = Azimuth(Az0, *params)
            Az1 = calc_angles(params[0],params[1])+90
            Abs1 = Azimuth(Az1, *params)
            Az2 = calc_angles(params[0],params[1])-90

            if alpha == 0:
                if Abs0 > Abs1:
                    Az[i,j] = Az0
                elif Abs1 > Abs0:                        
                    if Az1 < 90:
                        Az[i,j] = Az1
                    elif Az1 > 90:
                        Az[i,j] = Az2
            elif alpha == 90:
                if Abs0 < Abs1:
                    Az[i,j] = Az0
                elif Abs1 < Abs0:                        
                    if Az1 < 90:
                        Az[i,j] = Az1
                    elif Az1 > 90:
                        Az[i,j] = Az2    

            ampabs[i,j] = ampl1(*params)
            funcamp[i,j] = ampl2(params[0],params[1])
            Hermans[i,j] = OrFunc(alpha, *params)
            a0[i,j] = params[0]
            a1[i,j] = params[1]
            a2[i,j] = params[2]
            deg0[i,j] = images[0][i,j,wnidx]
            deg45[i,j] = images[1][i,j,wnidx]
            deg90[i,j] = images[2][i,j,wnidx]
            deg135[i,j] = images[3][i,j,wnidx]


    Az = np.reshape(Az, (Az.shape[0],Az.shape[1],1))
    Hermans = np.reshape(Hermans, (Hermans.shape[0],Hermans.shape[1],1))
    ampabs = np.reshape(ampabs, (ampabs.shape[0],ampabs.shape[1],1))
    funcamp = np.reshape(funcamp, (funcamp.shape[0],funcamp.shape[1],1))
    r_squared = np.reshape(r_squared, (r_squared.shape[0],r_squared.shape[1],1))
    a0 = np.reshape(a0, (a0.shape[0],a0.shape[1],1))
    a1 = np.reshape(a1, (a1.shape[0],a1.shape[1],1))
    a2 = np.reshape(a2, (a2.shape[0],a2.shape[1],1))
    deg0 = np.reshape(deg0, (deg0.shape[0],deg0.shape[1],1))
    deg45 = np.reshape(deg45, (deg45.shape[0],deg45.shape[1],1))
    deg90 = np.reshape(deg90, (deg90.shape[0],deg90.shape[1],1))
    deg135 = np.reshape(deg135, (deg135.shape[0],deg135.shape[1],1))

    return Az, Hermans, ampabs, funcamp, r_squared, a0, a1, a2, deg0, deg45, deg90, deg135



def process_polar_abs(images, wnidx, alpha, var, xy):


    hypercubes, lsx, lsy = get_hypercubes(images, xy)

    #wns = getx(images[0])
    wnidx = wnidx

    
    th, he, ampabs, funcamp, r2, a0, a1, a2, deg0, deg45, deg90, deg135 = compute(hypercubes, wnidx, alpha)

    var = [var]
    thr = np.radians(th)
    # join absorbance from images into a single image with a mean
    thrt = hypercube_to_table(thr, var, lsx, lsy)
    tht = hypercube_to_table(th, var, lsx, lsy)
    ampabst = hypercube_to_table(ampabs, var, lsx, lsy)
    funcampt = hypercube_to_table(funcamp, var, lsx, lsy)
    het = hypercube_to_table(he, var, lsx, lsy)
    r2t = hypercube_to_table(r2, var, lsx, lsy)

    output = ampabst
    output.th = thrt
    output.amp = het

    data = np.concatenate((th, he, ampabs, funcamp, r2), axis=2)    
    dom = ['Azimuth Angle', 'Hermans Orientation Function','Amplitude (Abs)','Function Amplitude','R-Squared of fitted cosine function']
    datat = hypercube_to_table(data, dom, lsx, lsy)

    model = np.concatenate((r2, a0, a1, a2, deg0, deg45, deg90, deg135), axis=2)
    modeldom = ['R-Squared of fitted cosine function','a0','a1','a2','0 degrees','45 degrees','90 degrees','135 degrees']
    modelt = hypercube_to_table(model, modeldom, lsx, lsy)


    return output, tht, ampabst, funcampt, het, r2t, datat, modelt

#calculate by "Stoke's Method"
def compute_stokes(images, wnidx):
    th = np.zeros(images[0].shape[:2])
    int = np.zeros(images[0].shape[:2])
    amp = np.zeros(images[0].shape[:2])
    for i in range(images[0].shape[0]):
        for j in range(images[0].shape[1]):
            temp = [images[0][i,j,wnidx],images[1][i,j,wnidx],images[2][i,j,wnidx],images[3][i,j,wnidx]]
            temp = np.reshape(np.asarray(temp,dtype=float),-1)

            th[i,j] = compute_theta(temp)
            int[i,j] = compute_intensity(temp)
            amp[i,j] = compute_amp(temp)

    th = np.reshape(th, (th.shape[0],th.shape[1],1))
    int = np.reshape(int, (int.shape[0],int.shape[1],1))
    amp = np.reshape(amp, (amp.shape[0],amp.shape[1],1))

    return th, int, amp
#Does not agree with other algorithm/published/reference data
def compute_theta(images):
    return 0.5 * np.arctan2(images[1] - images[3], images[0] - images[2])


def compute_intensity(images):
    S0 = (images[0] + images[1] + images[2] + images[3]) * 0.5
    return S0


def compute_amp(images):
    return np.sqrt((images[3] - images[1])**2 + (images[2] - images[0])**2) / compute_intensity(images)


def process_polar_stokes(images, wnidx, var, xy):
    hypercubes, lsx, lsy = get_hypercubes(images, xy)

    wnidx = wnidx
    wns = [var]

    thr, int, amp = compute_stokes(hypercubes, wnidx)
    th = np.degrees(thr)

    # join absorbance from images into a single image with a mean
    intensity = hypercube_to_table(int, wns, lsx, lsy)
    thrt = hypercube_to_table(thr, wns, lsx, lsy)
    tht = hypercube_to_table(th, wns, lsx, lsy)
    ampt = hypercube_to_table(amp, wns, lsx, lsy)

    data = np.concatenate((th,int,amp), axis=2)
    dom = ['Azimuth Angle','Amplitude','Orientation Function']
    datat = hypercube_to_table(data, dom, lsx, lsy)

    output = intensity
    output.th = thrt
    output.amp = ampt

    return output, intensity, tht, ampt, datat


def hypercube_to_table(hc, wns, lsx, lsy):
    table = build_spec_table(*_spectra_from_image(hc,
                             wns,
                             np.linspace(*lsx),
                             np.linspace(*lsy)))
    return table

class OWPolar(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Polar10"
    
    # Short widget description
    description = (
        "Callum's 4-angle polarisation implimentation with Stokes method")

    icon = "icons/unknown.svg"
    
    # Define inputs and outputs
    class Inputs:
        degree_sign = u"\N{DEGREE SIGN}"
        data = Input("Data", Orange.data.Table, default=True)        
        deg0 = Input(f"0{degree_sign}", Orange.data.Table)
        deg45 = Input(f"45{degree_sign}", Orange.data.Table)
        deg90 = Input(f"90{degree_sign}", Orange.data.Table)
        deg135 = Input(f"135{degree_sign}\-45{degree_sign}", Orange.data.Table)


    class Outputs:
        polar = Output("Polar Data", Orange.data.Table, default=True)
        combined = Output("Combined Polar Data",Orange.data.Table)
        model = Output("Curve Fit model data",Orange.data.Table)


    autocommit = settings.Setting(True)

    settingsHandler = DomainContextHandler()

    want_main_area = False
    resizing_enabled = False
    alpha = 0

    feature = ContextSetting(None)
    method = Setting(0)

    method_names = ('Curve Fitting Method','Stokes Method')

    class Warning(OWWidget.Warning):
        nodata = Msg("No useful data on input!")
        noang = Msg("Must receive 4 angles at specified polarisation")
        nofeat = Msg("Select Feature")
        wrongdata = Msg("Model returns inf. Inappropriate data")
        tomany = Msg("Widget must receive data at data input or discrete angles only")

    def __init__(self):
        super().__init__()

        self.data = None
        self.deg0 = None
        self.deg45 = None
        self.deg90 = None
        self.deg135 = None
        self.set_data(self.data)
        if self.data == None:
            self.set_data0(self.deg0)
            self.set_data45(self.deg45)
            self.set_data90(self.deg90)
            self.set_data135(self.deg135)

        dbox = gui.widgetBox(self.controlArea, "4-Angle Polarisation")

        ibox = gui.indentedBox(dbox)

        form = QWidget()
        formlayout = QFormLayout()
        form.setLayout(formlayout)
        ibox.layout().addWidget(form)

        box = gui.radioButtons(
            dbox, self, "method",
            callback=self._change_input)

        gui.widgetLabel(
            box, self.tr("Method to calculate 4-Angle Polarisation"))

        for opts in self.method_names:
            gui.appendRadioButton(box, self.tr(opts))

        self.featureselect = DomainModel(DomainModel.SEPARATED,
            valid_types=DomainModel.PRIMITIVE)
        self.group_view_x = gui.comboBox(
            ibox, self, "feature", searchable=True, box="Select feature",
            callback=self._change_input, model=self.featureselect)

        self.alphavalue = gui.lineEdit(ibox, self, "alpha", "Alpha value", callback=self.commit, valueType=int)

        #Commit
        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")
        self._change_input()
        self.contextAboutToBeOpened.connect(lambda x: self.init_attr_values(x[0]))

    #def wn_changed(self):
     #   self.wn    
     #  

    def init_attr_values(self, domain):
        self.featureselect.set_domain(domain)
        self.group_x = None
        self.group_y = None

    def _change_input(self):
        if self.method == 0:
            self.alphavalue.setDisabled(False)
        elif self.method == 1:
            self.alphavalue.setDisabled(True)
        self.commit()
        
    
    @Inputs.data
    def set_data(self, dataset):
        self.data = None
        self.Warning.nodata.clear()
        self.Warning.noang.clear()
        self.closeContext()
        self.data = dataset
        self.group = None
        if self.data is None:
            self.Warning.nodata()
        else:
            self.openContext(dataset.domain)
        self.Outputs.polar.send(None)
        self.Outputs.combined.send(None)
        self.Outputs.model.send(None)
        self.commit()

    @Inputs.deg0
    def set_data0(self, dataset):
        self.deg0 = None
        self.Warning.nodata.clear()
        self.Warning.noang.clear()
        self.closeContext()
        self.deg0 = dataset
        self.group = None
        if self.deg0 is None:
            self.Warning.noang()
        else:
            self.openContext(dataset.domain)
        self.Outputs.polar.send(None)
        self.Outputs.combined.send(None)
        self.Outputs.model.send(None)
        self.commit()

    @Inputs.deg45
    def set_data45(self, dataset):
        self.deg45 = None
        self.Warning.nodata.clear()
        self.Warning.noang.clear()
        self.closeContext()
        self.deg45 = dataset
        self.group = None
        if self.deg45 is None:
            self.Warning.noang()
        else:
            self.openContext(dataset.domain)
        self.Outputs.polar.send(None)
        self.Outputs.combined.send(None)
        self.Outputs.model.send(None)
        self.commit()

    @Inputs.deg90
    def set_data90(self, dataset):
        self.deg90 = None
        self.Warning.nodata.clear()
        self.Warning.noang.clear()
        self.closeContext()
        self.deg90 = dataset
        self.group = None
        if self.deg90 is None:
            self.Warning.noang()
        else:
            self.openContext(dataset.domain)
        self.Outputs.polar.send(None)
        self.Outputs.combined.send(None)
        self.Outputs.model.send(None)
        self.commit()

    @Inputs.deg135
    def set_data135(self, dataset):
        self.deg135 = None
        self.Warning.nodata.clear()
        self.Warning.noang.clear()
        self.closeContext()
        self.deg135 = dataset
        self.group = None
        if self.deg135 is None:
            self.Warning.noang()
        else:
            self.openContext(dataset.domain)
        self.Outputs.polar.send(None)
        self.Outputs.combined.send(None)
        self.Outputs.model.send(None)
        self.commit()

    def commit(self):
        if self.data and self.deg0 and self.deg45 and self.deg90 and self.deg135 is None:
            self.Outputs.polar.send(None)
            self.Outputs.combined.send(None)
            self.Outputs.model.send(None)
            return

        self.Warning.nofeat.clear()
        if self.feature is None:
            self.Warning.nofeat()
            return

        self.Warning.wrongdata.clear()
        self.Warning.tomany.clear()

        if self.deg0 and self.data is not None:
            self.Warning.tomany()
            return
        if self.deg45 and self.data is not None:
            self.Warning.tomany()
            return
        if self.deg90 and self.data is not None:
            self.Warning.tomany()
            return
        if self.deg135 and self.data is not None:
            self.Warning.tomany()
            return   

        # TODO for now this assumes images in the correct order of filenames with a Filename column
        if self.method == 0:
            if self.data is not None:

                fncol = self.data[:, "Filename"].metas.reshape(-1)
                unique_fns = np.unique(fncol)

                # split images into separate tables
                images = []
                for fn in unique_fns:
                    images.append(self.data[fn == fncol])

                # TODO align images according to their positions

                alpha = self.alpha
                xy = [self.data.domain[-1].name, self.data.domain[-2].name]
                wnidx = self.data.domain.index(f'{self.feature}')
                var = self.data.domain[wnidx].name

                out, tht, ampt, funcampt, het, r2t, comb, model = process_polar_abs(images, wnidx, alpha, var, xy)
                if np.isnan(comb.X).any():
                    self.Warning.wrongdata()
                else:
                    self.Outputs.polar.send(comb)
                    self.Outputs.combined.send(out)
                    self.Outputs.model.send(model)



            elif self.deg0 and self.deg45 and self.deg90 and self.deg135 is not None:

                alpha = self.alpha
                xy = [self.deg0.domain[-1].name, self.deg0.domain[-2].name]
                wnidx = self.deg0.domain.index(f'{self.feature}')
                var = self.deg0.domain[wnidx].name
                if var == self.deg45.domain[wnidx].name == self.deg90.domain[wnidx].name == self.deg135.domain[wnidx].name:
                    images = [self.deg0, self.deg45, self.deg90, self.deg135]

                    out, tht, ampt, funcampt, het, r2t, comb, model = process_polar_abs(images, wnidx, alpha, var, xy)
                    if np.isnan(comb.X).any():
                        self.Warning.wrongdata()
                    else:
                        self.Outputs.polar.send(comb)
                        self.Outputs.combined.send(out)
                        self.Outputs.model.send(model)


                

        elif self.method == 1:
            if self.data is not None:

                fncol = self.data[:, "Filename"].metas.reshape(-1)
                unique_fns = np.unique(fncol)

                # split images into separate tables
                images = []
                for fn in unique_fns:
                    images.append(self.data[fn == fncol])

                xy = [self.data.domain[-1].name, self.data.domain[-2].name]
                wnidx = self.data.domain.index(f'{self.feature}')
                var = self.data.domain[wnidx].name
                try:
                    out, int, th, amp, comb = process_polar_stokes(images, wnidx, var, xy)
                    self.Outputs.polar.send(comb)
                    self.Outputs.combined.send(out)
                    self.Outputs.model.send(None)
                except:
                    self.Warning.wrongdata()


            
            elif self.deg0 and self.deg45 and self.deg90 and self.deg135 is not None:

                xy = [self.deg0.domain[-1].name, self.deg0.domain[-2].name]
                wnidx = self.deg0.domain.index(f'{self.feature}')
                var = self.deg0.domain[wnidx].name
                if var == self.deg45.domain[wnidx].name == self.deg90.domain[wnidx].name == self.deg135.domain[wnidx].name:
                    images = [self.deg0, self.deg45, self.deg90, self.deg135]
                    try:
                        out, int, th, amp, comb = process_polar_stokes(images, wnidx, var, xy)
                        self.Outputs.polar.send(comb)
                        self.Outputs.combined.send(out)
                        self.Outputs.model.send(None)
                    except:
                        self.Warning.wrongdata()


                

            
        # TODO: 

if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPolar).run(Orange.data.Table("/home/marko/polar_preprocessed.pkl.gz"))
