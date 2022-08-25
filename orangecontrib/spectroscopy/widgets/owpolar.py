import os
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
from typing import List, Optional, Sequence

import math
from types import SimpleNamespace
import numpy as np
import pandas as pd
from AnyQt.QtCore import QItemSelectionModel, QItemSelection, QItemSelectionRange
from AnyQt.QtWidgets import QFormLayout, QWidget, QListView, QLabel, QSizePolicy
from scipy.optimize import curve_fit


import Orange.data
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Variable
from Orange.data.table import Table
from Orange.widgets.widget import OWWidget, Msg, Output, MultiInput
from Orange.widgets import gui, settings

from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.data import owconcatenate
from Orange.widgets.data.oweditdomain import disconnected
from orangewidget.utils.listview import ListViewSearch


def _restore_selected_items(model, view, setting, connector):
    selection = QItemSelection()
    sel_model: QItemSelectionModel = view.selectionModel()
    with disconnected(sel_model.selectionChanged,
                      connector):
        valid = []
        model_values = model[:]
        for var in setting:
            index = model_values.index(var)
            model_index = view.model().index(index, 0)
            selection.append(QItemSelectionRange(model_index))
            valid.append(var)
        sel_model.select(selection, QItemSelectionModel.ClearAndSelect)


class Results(SimpleNamespace):
    out = None
    model = None
    errorstate = 0

def sort_domain(domain):
    dom = [domain.metas, domain.attributes, domain.class_vars]
    sorted_dom_lst = []
    for i in dom:
        cvs = [[i, j, j.name] for i, j in enumerate(i)]
        rcvs_idx = []
        rcvs = []
        for j, k in enumerate(cvs):
            try:
                cvs[j][-1] = float(k[-1])
            except ValueError:
                rcvs.append(k[1])
                rcvs_idx.append(j)
        for j in reversed(rcvs_idx):
            cvs.pop(j)
        cvs_arr = np.array(cvs)
        if cvs_arr.shape[0] > 0:
            cvs_arr_sorted = cvs_arr[cvs_arr[:,2].argsort()]
            odom_cv = [i[1] for i in cvs_arr_sorted]
            odom = rcvs + odom_cv
        else:
            odom = rcvs
        sorted_dom_lst.append(tuple(odom))
    out = Domain(sorted_dom_lst[1], class_vars=sorted_dom_lst[2], metas=sorted_dom_lst[0])
    return out

def combine_visimg(data, polangles):
    atts = []
    for k, i in enumerate(data):
        try:
            temp = i.attributes['visible_images']
            for j in temp:
                tempname = str(j['name'] + f'({polangles[k]} Degrees)')
                dictcopy = j.copy()
                dictcopy.update({'name': tempname})
                atts = atts + [dictcopy]
        except KeyError:
            pass
        except AttributeError:
            pass
    attsdict = {'visible_images': atts}
    return attsdict

def run(data, feature, alpha, map_x, map_y, invert_angles, polangles, average,
        sep, state: TaskState):

    results = Results()

    output, model, spectra, origmetas, errorstate = process_polar_abs(data, alpha, feature, map_x,
                                                    map_y, invert_angles, polangles, average, state)

    tempoutaddmetas = [[ContinuousVariable.make('Azimuth Angle (' + i.name + ')'),
                    ContinuousVariable.make('Hermans Orientation Function (' + i.name + ')'),
                    ContinuousVariable.make('Intensity (' + i.name + ')'),
                    ContinuousVariable.make('Amplitude (' + i.name + ')'),
                    ContinuousVariable.make('R-squared (' + i.name + ')')] for i in feature]
    outaddmetas = []
    for i in tempoutaddmetas:
        outaddmetas = outaddmetas + i

    tempmodaddmetas = [[ContinuousVariable.make('R-squared (' + i.name + ')'),
                    ContinuousVariable.make('a0 (' + i.name + ')'),
                    ContinuousVariable.make('a1 (' + i.name + ')'),
                    ContinuousVariable.make('a2 (' + i.name + ')')] for i in feature]
    modaddmetas = []
    for i in tempmodaddmetas:
        modaddmetas = modaddmetas + i
    ometadom = data[0].domain.metas
    if average is False:
        values = tuple(f'{i} Degrees' for i in polangles)
        PolAng = DiscreteVariable.make('Polarisation Angle', values=values)
        outmetadom = (ometadom + tuple([PolAng]) + tuple(outaddmetas))
        modmetadom = (ometadom + tuple([PolAng]) + tuple(modaddmetas))
        output_stack = tuple(output for i in polangles)
        model_stack = tuple(model for i in polangles)
        output = np.vstack(output_stack)
        model = np.vstack(model_stack)
    elif average is True:
        if sep is not None:
            sep_idx = ometadom.index(sep)
            try:
                origmetas = np.c_[origmetas[:,0:sep_idx], origmetas[:,sep_idx+1:]]
            except ValueError:
                origmetas = np.r_[origmetas[:,0:sep_idx]]
            ometadom = tuple(i for i in ometadom if i is not sep)
        outmetadom = (ometadom + tuple(outaddmetas))
        modmetadom = (ometadom + tuple(modaddmetas))

    ofeatdom = data[0].domain.attributes
    datadomain = Domain(ofeatdom, metas = outmetadom)
    moddomain = Domain(ofeatdom, metas = modmetadom)
    outmetas = np.hstack((origmetas, output))
    modmetas = np.hstack((origmetas, model))

    out = Table.from_numpy(datadomain, X=spectra, Y=None, metas=outmetas)
    mod = Table.from_numpy(moddomain, X=spectra, Y=None, metas=modmetas)

    results.out = out
    results.model = mod
    results.errorstate = errorstate

    attsdict = combine_visimg(data, polangles)

    results.out.attributes = attsdict
    results.model.attributes = attsdict
    return results

#Calculate by fitting to function
def azimuth(x,a0,a1,a2):
    t = 2*np.radians(x)
    return a0*np.sin(t)+a1*np.cos(t)+a2

def azimuth_jac(x, a0, a1, a2):
    t = 2*np.radians(x).reshape(-1, 1)
    da0 = np.sin(t)
    da1 = np.cos(t)
    da2 = np.ones(t.shape)
    return np.hstack((da0, da1, da2))

def calc_angles(a0,a1):
    return np.degrees(0.5*np.arctan(a0/a1))

def ampl1(a0,a1,a2):
    return (a2+(math.sqrt(a0**2+a1**2))+a2-(math.sqrt(a0**2+a1**2)))

def ampl2(a0,a1):
    return (2*(math.sqrt(a0**2+a1**2)))

def orfunc(alpha,a0,a1,a2):
    if alpha < 54.73:
        Dmax = (2*a2+2*math.sqrt(a0**2+a1**2))/(2*a2-2*math.sqrt(a0**2+a1**2))
        return ((Dmax-1)/(Dmax+2)*(2/(3*np.cos(np.radians(alpha))**2-1)))
    elif alpha >= 54.73:
        Dmin = (2*a2-2*math.sqrt(a0**2+a1**2))/(2*a2+2*math.sqrt(a0**2+a1**2))
        return ((Dmin-1)/(Dmin+2)*(2/(3*np.cos(np.radians(alpha))**2-1)))
    return None

def find_az(alpha, params):
    Az0 = calc_angles(params[0],params[1])
    Abs0 = azimuth(Az0, *params)
    Az1 = calc_angles(params[0],params[1])+90
    Abs1 = azimuth(Az1, *params)
    Az2 = calc_angles(params[0],params[1])-90

    if alpha < 54.73:
        if Abs0 > Abs1:
            Az = Az0
        elif Abs1 > Abs0:
            if Az1 < 90:
                Az = Az1
            elif Az1 > 90:
                Az = Az2
    elif alpha >= 54.73:
        if Abs0 < Abs1:
            Az = Az0
        elif Abs1 < Abs0:
            if Az1 < 90:
                Az = Az1
            elif Az1 > 90:
                Az = Az2
    return Az

def compute(xys, yidx, smms, shapes, dtypes, polangles):
    tcvs = smms[0]
    cvs = np.ndarray(shapes[0], dtype=dtypes[0], buffer=tcvs.buf)
    tout = smms[3]
    out = np.ndarray(shapes[3], dtype=dtypes[3], buffer=tout.buf)
    tmod = smms[4]
    mod = np.ndarray(shapes[4], dtype=dtypes[4], buffer=tmod.buf)
    tcoords = smms[5]
    coords = np.ndarray(shapes[5], dtype=dtypes[5], buffer=tcoords.buf)
    tvars = smms[6]
    vars = np.ndarray(shapes[6], dtype=dtypes[6], buffer=tvars.buf)

    x = np.asarray(polangles)

    for i in range(yidx[0], yidx[1]):#y-values(rows)
        if vars[1] == 1:
            break
        for j in enumerate(xys[0]):#x-values(cols)
            for l in range(cvs.shape[2]):
                if np.any(np.isnan(cvs[i,j[0],l,:]), axis=0):
                    continue
                out[i,j[0],l,0] = coords[i,j[0],1]#x-map
                mod[i,j[0],l,0] = coords[i,j[0],1]
                out[i,j[0],l,1] = coords[i,j[0],0]#y-map
                mod[i,j[0],l,1] = coords[i,j[0],0]

                temp = list(cvs[i,j[0],l,:])

                params = curve_fit(azimuth, x, temp, jac=azimuth_jac)[0]

                residuals = temp - azimuth(x, *params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((temp-np.mean(temp))**2)
                if ss_tot == 0:
                    vars[1] = 1
                    break
                out[i,j[0],l,6] = 1-(ss_res/ss_tot)
                mod[i,j[0],l,2] = 1-(ss_res/ss_tot)
                out[i,j[0],l,2] = find_az(vars[0], params)
                out[i,j[0],l,3] = orfunc(vars[0], *params)
                out[i,j[0],l,4] = ampl1(*params)
                out[i,j[0],l,5] = ampl2(params[0],params[1])
                mod[i,j[0],l,3] = params[0]
                mod[i,j[0],l,4] = params[1]
                mod[i,j[0],l,5] = params[2]


def unique_xys(images, map_x, map_y):
    lsxs = np.empty(0)
    lsys = np.empty(0)
    for i in enumerate(images):
        tempdata = i[1].transform(Domain([map_x, map_y]))
        lsx = np.unique(tempdata.X[:,0])
        lsy = np.unique(tempdata.X[:,1])
        lsxs = np.append(lsxs, lsx)
        lsys = np.append(lsys, lsy)

    ulsxs = np.unique(lsxs)
    ulsys = np.unique(lsys)
    return ulsxs, ulsys

def start_compute(ulsxs, ulsys, names, shapes, dtypes, polangles, state):
    # single core processing is faster for small data sets and small number of selected features
    # if <data size> > x:
    ncpu = os.cpu_count()
    # ncpu = 6
    tulsys = np.array_split(ulsys, ncpu)
    state.set_status("Calculating...")
    threads=[]
    cumu = 0
    for i in range(ncpu):
        tlsxys = [ulsxs,tulsys[i]]
        yidx = [cumu, cumu+len(tulsys[i])]
        cumu += len(tulsys[i])
        # compute(tlsxys, yidx, shapes, dtypes, polangles, i)
        t = multiprocessing.Process(target=compute,
                                    args=(tlsxys, yidx, names, shapes, dtypes, polangles))
        threads.append(t)
        t.start()

    # for t in threads:
    #     t.join()

    # else:
        # ncpu = 1
        # tulsys = np.array_split(ulsys, ncpu)
        # state.set_status("Calculating...")
        # threads=[]
        # cumu = 0
        # for i in range(ncpu):
        #     tlsxys = [ulsxs,tulsys[i]]
        #     yidx = [cumu, cumu+len(tulsys[i])]
        #     cumu += len(tulsys[i])
        #     compute(tlsxys, yidx, shapes, dtypes, polangles, i)
    return threads

def process_polar_abs(images, alpha, feature, map_x, map_y, invert, polangles, average, state):
    state.set_status("Preparing...")

    ulsxs, ulsys = unique_xys(images, map_x, map_y)

    if len(ulsxs) > 1:
        dx = np.sum(np.diff(ulsxs))/(len(ulsxs)-1)
    else:
        dx = 1
    if len(ulsys) > 1:
        dy = np.sum(np.diff(ulsys))/(len(ulsys)-1)
    else:
        dy = 1
    minx = np.min(ulsxs)
    miny = np.min(ulsys)

    featnames = [i.name for i in feature]
    cvs = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], len(featnames), len(images)), np.nan)
    spec = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0],
                    images[0].X.shape[1], len(images)), np.nan, dtype=object)
    metas = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0],
                     images[0].metas.shape[1], len(images)), np.nan, dtype=object)
    out = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], len(featnames), 7), np.nan)
    mod = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], len(featnames), 6), np.nan)
    coords = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], 2), np.nan)
    vars = np.asarray([alpha, 0])
    fill = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0]), np.nan)
    for i, j in enumerate(images):
        cv = [j.domain[k] for k in featnames]
        doms = [map_x, map_y] + cv
        tempdata: Table = j.transform(Domain(doms))
        temp_xy = tempdata.X[:,0:2].copy()
        temp_xy[:,0] = np.rint(((temp_xy[:,0]-minx)/dx))
        temp_xy[:,1] = np.rint(((temp_xy[:,1]-miny)/dy))
        temp_xy = np.array(temp_xy, dtype=np.int_)
        cvs[temp_xy[:,1],temp_xy[:,0],:,i] = tempdata[:,2:]
        spec[temp_xy[:,1],temp_xy[:,0],:,i] = j.X
        metas[temp_xy[:,1],temp_xy[:,0],:,i] = j.metas
    xys = pd.DataFrame(fill, index=ulsys, columns=ulsxs, dtype=object)
    for k, i in enumerate(xys.index):
        for l, j in enumerate(xys.columns):
            coords[k,l,0] = i
            coords[k,l,1] = j

    with SharedMemoryManager() as smm:
        tcvs = smm.SharedMemory(size=cvs.nbytes)
        scvs = np.ndarray(cvs.shape, dtype=cvs.dtype, buffer=tcvs.buf)
        scvs[:,:,:] = cvs[:,:,:]
        tout = smm.SharedMemory(size=out.nbytes)
        sout = np.ndarray(out.shape, dtype=out.dtype, buffer=tout.buf)
        sout[:,:,:,:] = out[:,:,:,:]
        tmod = smm.SharedMemory(size=mod.nbytes)
        smod = np.ndarray(mod.shape, dtype=mod.dtype, buffer=tmod.buf)
        smod[:,:,:,:] = mod[:,:,:,:]
        tcoords = smm.SharedMemory(size=coords.nbytes)
        scoords = np.ndarray(coords.shape, dtype=coords.dtype, buffer=tcoords.buf)
        scoords[:,:,:] = coords[:,:,:]
        tvars = smm.SharedMemory(size=vars.nbytes)
        svars = np.ndarray(vars.shape, dtype=vars.dtype, buffer=tvars.buf)
        svars[:] = vars[:]

        smms = [tcvs, None, None, tout, tmod, tcoords, tvars]
        shapes = [cvs.shape, spec.shape, metas.shape, out.shape, mod.shape, coords.shape, vars.shape]
        dtypes = [cvs.dtype, spec.dtype, metas.dtype, out.dtype, mod.dtype, coords.dtype, vars.dtype]

        threads = start_compute(ulsxs, ulsys, smms, shapes, dtypes, polangles, state)

        for t in threads:
            t.join()

        state.set_status("Finishing...")
        if invert is True:
            sout[:,:,:,2] = sout[:,:,:,2]*-1
        outputs = np.reshape(sout[:,:,:,2:], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], 5*len(featnames)))
        model = np.reshape(smod[:,:,:,2:], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], 4*len(featnames)))

    spectra = []
    met = []
    if average is False:
        for i in range(len(polangles)):
            spectratemp = np.reshape(spec[:,:,:,i],
                                    (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].X.shape[1]))
            spectratemp = spectratemp[~np.isnan(model).any(axis=1)]
            spectra.append(spectratemp)
            metatemp = np.reshape(metas[:,:,:,i],
                                (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].metas.shape[1]))
            metatemp = metatemp[~np.isnan(model).any(axis=1)]
            metatemp = np.append(metatemp, np.full((np.shape(metatemp)[0],1), i), axis=1)
            met.append(metatemp)
    elif average is True:
        average_spec = np.average(spec, axis=3)
        spectratemp = np.reshape(average_spec,
                                (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].X.shape[1]))
        spectratemp = spectratemp[~np.isnan(model).any(axis=1)]
        spectra.append(spectratemp)
        metatemp = np.reshape(metas[:,:,:,0],
                            (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].metas.shape[1]))
        metatemp = metatemp[~np.isnan(model).any(axis=1)]
        met.append(metatemp)

    outputs = outputs[~np.isnan(model).any(axis=1)]
    model = model[~np.isnan(model).any(axis=1)]

    spectra = np.concatenate((spectra), axis=0)
    meta = np.concatenate((met), axis=0)

    return outputs, model, spectra, meta, vars[1]


class OWPolar(OWWidget, ConcurrentWidgetMixin):

    # Widget's name as displayed in the canvas
    name = "4+ Angle Polarisation"

    # Short widget description
    description = (
        "Calculate Azimuth Angle, Orientation function, Amplitude and Intensity of "
        "vibrational mode(s) using polarised data measured at 4 or more polarisation angles.")

    icon = "icons/polar.svg"

    # Define inputs and outputs
    class Inputs:
        data = MultiInput("Data", Orange.data.Table, default=True)

    class Outputs:
        polar = Output("Polar Data", Orange.data.Table, default=True)
        model = Output("Curve Fit model data", Orange.data.Table)

    autocommit = settings.Setting(False)

    settingsHandler = DomainContextHandler()

    want_main_area = False
    resizing_enabled = True
    alpha = Setting(0, schema_only=True)

    map_x = ContextSetting(None)
    map_y = ContextSetting(None)
    invert_angles = Setting(False, schema_only=True)
    average = Setting(False, schema_only=True)
    angles = ContextSetting(None)

    anglst = []
    lines = []
    labels = []
    multiin_anglst = []
    multiin_lines = []
    multiin_labels = []
    polangles = []
    n_inputs = 0

    feats: List[Variable] = ContextSetting([])

    class Warning(OWWidget.Warning):
        nodata = Msg("No useful data on input!")
        noang = Msg("Must receive 4 angles at specified polarisation")
        nofeat = Msg("Select Feature")
        noxy = Msg("Select X and Y variables")
        pol = Msg("Invalid Polarisation angles")
        notenough = Msg("Must have >= 4 angles")
        wrongdata = Msg("Model returns inf. Inappropriate data")
        tomany = Msg("Widget must receive data at data input or discrete angles only")
        missingfeat = Msg("All inputs must have the selected feature")
        renamed_variables = Msg("Variables with duplicated names have been renamed.")
        XYfeat = Msg("Selected feature(s) cannot be the same as XY selection")

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        gui.OWComponent.__init__(self)

        self._dumb_tables = owconcatenate.OWConcatenate._dumb_tables
        self._get_part = owconcatenate.OWConcatenate._get_part
        self.merge_domains = owconcatenate.OWConcatenate.merge_domains

        self._data_inputs: List[Optional[Table]] = []

        hbox = gui.hBox(self.controlArea)
        #col 1

        vbox2 = gui.vBox(hbox, "Inputs")

        form2 = QWidget()
        formlayout2 = QFormLayout()
        form2.setLayout(formlayout2)

        self.multifile = gui.widgetBox(vbox2, "Multifile Input (all angles in 1 table)",
                                       sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.anglemetas = DomainModel(DomainModel.METAS, valid_types=DiscreteVariable)
        self.anglesel = gui.comboBox(self.multifile, self, 'angles', searchable=True,
                                     label='Select Angles by:', callback=self._change_angles,
                                     model=self.anglemetas)
        self.anglesel.setDisabled(True)


        self.multiin = gui.widgetBox(vbox2, "Multiple Inputs (1 angle per input)",
                                     sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Fixed))

        vbox2.layout().addWidget(form2)

        #col 2
        vbox1 = gui.vBox(hbox, "Features")

        self.featureselect = DomainModel(DomainModel.SEPARATED,
            valid_types=ContinuousVariable)
        self.feat_view = ListViewSearch(selectionMode=QListView.ExtendedSelection)
        self.feat_view.setModel(self.featureselect)
        self.feat_view.selectionModel().selectionChanged.connect(self._feat_changed)
        vbox1.layout().addWidget(self.feat_view)
        vbox1.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum))
        self.contextOpened.connect(
            lambda: _restore_selected_items(model=self.featureselect,
                                            view=self.feat_view,
                                            setting=self.feats,
                                            connector=self._feat_changed))

        #col 3
        vbox = gui.vBox(hbox, "Parameters")

        form = QWidget()
        formlayout = QFormLayout()
        form.setLayout(formlayout)

        xybox = gui.widgetBox(vbox, "Data XY Selection",
                              sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.x_axis = DomainModel(DomainModel.METAS, valid_types=ContinuousVariable)
        self.y_axis = DomainModel(DomainModel.METAS, valid_types=ContinuousVariable)

        gui.comboBox(xybox, self, 'map_x', searchable=True, label="X Axis",
            callback=self._change_input, model=self.x_axis)
        gui.comboBox(xybox, self, 'map_y', searchable=True, label="Y Axis",
            callback=self._change_input, model=self.y_axis)

        vbox.layout().addWidget(form)

        pbox = gui.widgetBox(vbox, sizePolicy=(QSizePolicy.Minimum, QSizePolicy.Fixed))
        gui.lineEdit(pbox, self, "alpha", "Alpha value",
                                       callback=self._change_input, valueType=int)

        gui.checkBox(pbox, self, 'invert_angles', label="Invert Angles",
                     callback=self._change_input)

        gui.checkBox(pbox, self, 'average', label='Average Spectra',
                     callback=self._change_input)

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply", commit=self.commit)
        self._change_input()
        self.contextAboutToBeOpened.connect(lambda x: self.init_attr_values(x[0]))


    def _feat_changed(self):
        rows = self.feat_view.selectionModel().selectedRows()
        values = self.feat_view.model()[:]
        self.feats = [values[row.row()] for row in sorted(rows)]
        self.commit.deferred()

    def init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.featureselect.set_domain(domain)
        self.x_axis.set_domain(domain)
        self.y_axis.set_domain(domain)
        self.anglemetas.set_domain(domain)

    def _change_input(self):
        self.commit.deferred()

    def _change_angles(self):
        self.Warning.nodata.clear()
        if self.angles:
            self.clear_angles(self.anglst, self.lines, self.labels, self.multifile)
            self.anglst = []
            self.lines = []
            self.labels = []
            self.Warning.notenough.clear()
            if len(self.angles.values) < 4:
                self.Warning.notenough()
            else:
                tempangles = np.linspace(0, 180, len(self.angles.values)+1)
                for i, j in enumerate(self.angles.values):
                    self.add_angles(self.anglst, j, self.labels, self.lines, self.multifile,
                                    i, tempangles[i], self._send_angles)
                self._send_angles()
                for i in self.labels:
                    i.setDisabled(False)
                for i in self.lines:
                    i.setDisabled(False)
                self.commit.deferred()

    def add_angles(self, anglst, lab, labels, lines, widget,
                   i, place, callback): #to be used in a loop
        file = os.path.basename(lab)
        anglst.append(lab)
        ledit = gui.lineEdit(widget, self, None, label = file, callback = callback)
        ledit.setText(str(place))
        lines.append(ledit)
        for j in ledit.parent().children():
            if isinstance(j, QLabel):
                labels.append(j)

    def clear_angles(self, anglst, lines, labels, widget):
        if widget is self.multiin:
            for i in reversed(range(self.multiin.layout().count())):
                self.multiin.layout().itemAt(i).widget().setParent(None)
        if widget is self.multifile:
            for i in reversed(range(self.multifile.layout().count())):
                if i != 0:
                    self.multifile.layout().itemAt(i).widget().setParent(None)
        anglst.clear()
        lines.clear()
        labels.clear()
        self.polangles.clear()

    def _send_ind_angles(self):
        self.polangles.clear()
        for i in self.multiin_lines:
            self.polangles.append(i.text())
        try:
            pol = []
            for i in self.polangles:
                pol.append(float(i))
            self.polangles = pol
            self.commit.deferred()
        except ValueError:
            pass

    def _send_angles(self):
        self.polangles.clear()
        for i in self.lines:
            self.polangles.append(i.text())
        try:
            pol = []
            for i in self.polangles:
                pol.append(float(i))
            self.polangles = pol
            self.commit.deferred()
        except ValueError:
            pass

    def input_select(self):
        self.angles = None
        if len(self.data) == 0 or 1 < len(self.data) < 4:
            self.anglesel.setDisabled(True)
            for i in self.multiin_labels:
                i.setDisabled(True)
            for i in self.multiin_lines:
                i.setDisabled(True)
        elif len(self.data) == 1:
            self.anglesel.setDisabled(False)
            for i in self.multiin_labels:
                i.setDisabled(True)
            for i in self.multiin_lines:
                i.setDisabled(True)
        elif len(self.data) > 3:
            self.anglesel.setDisabled(True)
            for i in self.multiin_labels:
                i.setDisabled(False)
            for i in self.multiin_lines:
                i.setDisabled(False)
            self._send_ind_angles()

    def check_params(self):
        self.Warning.nofeat.clear()
        if self.feats is None or len(self.feats) == 0:
            self.Warning.nofeat()
            return
        self.Warning.noxy.clear()
        if self.map_x is None or self.map_y is None:
            self.Warning.noxy()
            return
        self.Warning.pol.clear()
        if len(self.polangles) == 0:
            self.Warning.pol()
            return
        for i in self.polangles:
            if isinstance(i, float) is False:
                self.Warning.pol()
                return
        self.Warning.XYfeat.clear()
        for i in self.feats:
            if i in (self.map_x, self.map_y):
                self.Warning.XYfeat()
                return
        self.Warning.wrongdata.clear()

    @Inputs.data
    def set_data(self, index: int, dataset: Table):
        self._data_inputs[index] = dataset

    @Inputs.data.insert
    def insert_data(self, index, dataset):
        self._data_inputs.insert(index, dataset)
        self.n_inputs += 1
        self.idx = index

    @Inputs.data.remove
    def remove_data(self, index):
        self._data_inputs.pop(index)
        self.n_inputs -= 1
        self.polangles.clear()

    @property
    def more_data(self) -> Sequence[Table]:
        return [t for t in self._data_inputs if t is not None]

    def handleNewSignals(self):
        self.check_and_close_context()
        self.data = None
        self.Warning.clear()
        self.Outputs.polar.send(None)
        self.Outputs.model.send(None)
        self.data = self.more_data

        self.clear_angles(self.anglst, self.lines, self.labels, self.multifile)
        self.clear_angles(self.multiin_anglst, self.multiin_lines,
                          self.multiin_labels, self.multiin)

        names = [i.name for i in self.data]

        tempangles = np.linspace(0, 180, len(self.data)+1)
        for i in range(len(self.data)):
            self.add_angles(self.multiin_anglst, names[i], self.multiin_labels,
                            self.multiin_lines, self.multiin, i, tempangles[i],
                            self._send_ind_angles)

        self.input_select()

        if len(self.data) == 0:
            self.Outputs.polar.send(None)
            self.Outputs.model.send(None)
            self.contextAboutToBeOpened.emit([Table.from_domain(Domain(()))])
            return

        if len(self.data) == 1:
            self.openContext(self.data[0])
            self._change_angles()
        elif 1 < len(self.data) < 4 or len(self.data) == 0:
            self.Warning.notenough()
            self.contextAboutToBeOpened.emit([Table.from_domain(Domain(()))])
            return
        else:
            tables = self._dumb_tables(self)
            domains = [table.domain for table in tables]
            self._get_part = self._get_part
            self.merge_type = 0
            domain1 = self.merge_domains(self, domains)
            domain1 = sort_domain(domain1)
            self.merge_type = 1
            domain2 = self.merge_domains(self, domains)

            self.sorted_data = [table.transform(domain1) for table in tables]
            self.openContext(Table.from_domain(domain2))

        self.commit.now()

    @gui.deferred
    def commit(self):

        self.check_params()
        if len(self.Warning.active) > 0:
            return

        if len(self.data) == 1:
            if self.angles:
                fncol = self.data[0][:, self.angles.name].metas.reshape(-1)
                images = []
                for fn in self.anglst:
                    images.append(self.data[0][self.angles.to_val(fn) == fncol])
                sorted_data = images
            else:
                return
        elif 1 < len(self.data) < 4:
            self.Warning.notenough()
            self.Outputs.polar.send(None)
            self.Outputs.model.send(None)
            return
        else:
            sorted_data = self.sorted_data

        self.setInvalidated(True)
        self.setReady(False)

        self.start(run, sorted_data, self.feats, self.alpha, self.map_x,
                   self.map_y, self.invert_angles, self.polangles,
                   self.average, self.angles)

    def on_done(self, result: Results):
        self.setInvalidated(False)
        self.setReady(True)
        if result is None:
            self.Outputs.polar.send(None)
            self.Outputs.model.send(None)
            return
        if result.errorstate == 1:
            self.Warning.wrongdata()
        else:
            self.Outputs.polar.send(result.out)
            self.Outputs.model.send(result.model)

    def on_partial_result(self, result):
        self.setInvalidated(False)
        self.setReady(True)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def check_and_close_context(self):
        if hasattr(self, 'data'):
            if len(self.data) == 1:
                self.closeContext()
            else:
                self.angles = None
                self.closeContext()


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    import orangecontrib.spectroscopy  # so that can be loaded
    WidgetPreview(OWPolar).run(insert_data=[(0, Orange.data.Table("polar/4-angle-ftir_multifile.tab"))])
