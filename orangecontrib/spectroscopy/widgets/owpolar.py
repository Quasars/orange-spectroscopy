import warnings
import os
import multiprocessing
from multiprocessing import shared_memory
from typing import List, Optional, Sequence
from collections import OrderedDict

from Orange.util import wrap_callback
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from PyQt5.QtWidgets import QGridLayout
from scipy.optimize import OptimizeWarning

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minpack
import math
from types import SimpleNamespace

from orangecanvas.scheme import scheme
from orangecanvas.scheme import SchemeNode
from orangewidget.workflow.widgetsscheme import WidgetsScheme

import Orange.data
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Variable
from Orange.data.table import Table
from Orange.widgets.widget import OWWidget, Msg, Input, Output, MultiInput
from Orange.widgets import gui, settings
from orangewidget.gui import LineEditWFocusOut

from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler, SettingProvider
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatOrNone
from Orange.widgets.utils.itemmodels import DomainModel, VariableListModel
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.annotated_data import add_columns
from orangewidget.utils.listview import ListViewSearch
from Orange.data.util import get_indices

from AnyQt.QtWidgets import QFormLayout, QWidget, QSplitter, QListView, QLabel, QLineEdit, QAbstractItemView, QSizePolicy
from AnyQt.QtCore import Qt, QSize

from orangecontrib.spectroscopy.data import _spectra_from_image, getx, build_spec_table
from orangecontrib.spectroscopy.utils import get_hypercube

import time

class Results(SimpleNamespace):
    out = None
    model = None
    errorstate = 0

def unify_tables(piece):
    if len(piece) > 0:
        dom_n = [i.name for i in piece]
        un, n = np.unique(dom_n, return_index=True)
        dom = np.full((max(n)+1), None, dtype=object)
        for i in n:
            dom[i] = piece[i] 
        odom = [i for i in dom if i != None]        
        return odom
    else:
        return None
    
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
        except:
            pass
    attsdict = {'visible_images': atts}
    return attsdict

def run0(data, feature, alpha, map_x, map_y, invert_angles, polangles, state: TaskState):

    results = Results()
        
    alpha = alpha
    
    output, model, spectra, origmetas, errorstate = process_polar_abs(data, alpha, feature, map_x, map_y, invert_angles, polangles, state)    
    
    
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
    values = tuple([f'{i} Degrees' for i in polangles])
    PolAng = DiscreteVariable.make('Polarisation Angle', values=values)      
    
    ometadom = data[0].domain.metas
    outmetadom = (ometadom + tuple([PolAng]) + tuple(outaddmetas))
    modmetadom = (ometadom + tuple([PolAng]) + tuple(modaddmetas))
    ofeatdom = data[0].domain.attributes
    datadomain = Domain(ofeatdom, metas = outmetadom)
    moddomain = Domain(ofeatdom, metas = modmetadom)

    output_stack = tuple([output for i in polangles])
    model_stack = tuple([model for i in polangles])
    output = np.vstack(output_stack)
    model = np.vstack(model_stack)

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
 
# def run1(data, feature, map_x, map_y, invert_angles, state: TaskState):
#     results = Results()
#     if data is not None:
#         fncol = data[:, "Filename"].metas.reshape(-1)
#         unique_fns = np.unique(fncol)

#         # split images into separate tables
#         images = []
#         for fn in unique_fns:
#             images.append(data[fn == fncol])

#         try:
#             output, spectra, origmetas = process_polar_stokes(images, feature, map_x, map_y, invert_angles, state)
            
#             tempoutaddmetas = [[ContinuousVariable.make('Azimuth Angle (' + i.name + ')'),
#                             ContinuousVariable.make('Amplitude (' + i.name + ')'),
#                             ContinuousVariable.make('Intensity (' + i.name + ')'),] for i in feature]
#             outaddmetas = []
#             for i in tempoutaddmetas:
#                 outaddmetas = outaddmetas + i

#             PolAng = DiscreteVariable.make('Polarisation Angle', values=('0 Degrees','45 Degrees','90 Degrees','135/-45 Degrees'))

#             ometadom = data.domain.metas
#             outmetadom = (ometadom + tuple([PolAng]) + tuple(outaddmetas))
#             ofeatdom = data.domain.attributes
#             datadomain = Domain(ofeatdom, metas = outmetadom)

#             output = np.vstack((output, output, output, output))

#             outmetas = np.hstack((origmetas, output))

#             out = Table.from_numpy(datadomain, X=spectra, Y=None, metas=outmetas)

#             results.out = out
#             results.out.attributes = data.attributes
#             return results
#         except:
#             OWPolar.Warning.wrongdata()
    
    # elif deg0 and deg45 and deg90 and deg135 is not None:
        
    #     for i in feature:
    #         featname = i.name
    #         try:
    #             deg0.domain[featname]
    #             deg45.domain[featname]
    #             deg90.domain[featname]
    #             deg135.domain[featname]
    #         except:
    #             OWPolar.Warning.missingfeat()
    #             return results

    #     images = [deg0, deg45, deg90, deg135]

    #     metas = []
    #     attrs = []
    #     class_vars = []
    #     for i in images:
    #         metas = metas+[j for j in i.domain.metas]
    #         attrs = attrs+[j for j in i.domain.attributes]
    #         class_vars = class_vars+[j for j in i.domain.class_vars]

    #     attrs = unify_tables(attrs)
    #     metas = unify_tables(metas)
    #     class_vars = unify_tables(class_vars)

    #     dom = Domain(attrs, metas = metas)

    #     for i, j in enumerate(images):
    #         images[i] = j.transform(dom)

    #     try:
    #         output, spectra, origmetas = process_polar_stokes(images, feature, map_x, map_y, invert_angles, state)

    #         tempoutaddmetas = [[ContinuousVariable.make('Azimuth Angle (' + i.name + ')'),
    #                         ContinuousVariable.make('Amplitude (' + i.name + ')'),
    #                         ContinuousVariable.make('Intensity (' + i.name + ')'),] for i in feature]
    #         outaddmetas = []
    #         for i in tempoutaddmetas:
    #             outaddmetas = outaddmetas + i

    #         PolAng = DiscreteVariable.make('Polarisation Angle', values=('0 Degrees','45 Degrees','90 Degrees','135/-45 Degrees'))
    #         #TODO: ensure all inputs have same domain variables/concatenate all domain vars
    #         ometadom = images[0].domain.metas
    #         outmetadom = (ometadom + tuple([PolAng]) + tuple(outaddmetas))
    #         ofeatdom = images[0].domain.attributes
    #         datadomain = Domain(ofeatdom, metas = outmetadom)

    #         output = np.vstack((output, output, output, output))

    #         outmetas = np.hstack((origmetas, output))

    #         out = Table.from_numpy(datadomain, X=spectra, Y=None, metas=outmetas)

    #         results.out = out
    #         results.out.attributes = deg0.attributes
    #         return results
    #     except:
    #         OWPolar.Warning.wrongdata()       

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
    if alpha < 54.73:
        Dmax = (2*a2+2*math.sqrt(a0**2+a1**2))/(2*a2-2*math.sqrt(a0**2+a1**2))
        return ((Dmax-1)/(Dmax+2)*(2/(3*np.cos(np.radians(alpha))**2-1)))
    elif alpha >= 54.73:
        Dmin = (2*a2-2*math.sqrt(a0**2+a1**2))/(2*a2+2*math.sqrt(a0**2+a1**2))
        return ((Dmin-1)/(Dmin+2)*(2/(3*np.cos(np.radians(alpha))**2-1)))

def compute(xys, yidx, shapes, dtypes, polangles):

    tcvs = shared_memory.SharedMemory(name='cvs', create=False)
    cvs = np.ndarray(shapes[0], dtype=dtypes[0], buffer=tcvs.buf)
    tout = shared_memory.SharedMemory(name='out', create=False)
    out = np.ndarray(shapes[3], dtype=dtypes[3], buffer=tout.buf)  
    tmod = shared_memory.SharedMemory(name='mod', create=False)
    mod = np.ndarray(shapes[4], dtype=dtypes[4], buffer=tmod.buf)
    tcoords = shared_memory.SharedMemory(name='coords', create=False)
    coords = np.ndarray(shapes[5], dtype=dtypes[5], buffer=tcoords.buf)
    tvars = shared_memory.SharedMemory(name='vars', create=False)
    vars = np.ndarray(shapes[6], dtype=dtypes[6], buffer=tvars.buf)
    
    x = np.asarray(polangles)  

    for i in range(yidx[0], yidx[1]):#y-values(rows)
        if vars[1] == 1:
            break
        for j, k in enumerate(xys[0]):#x-values(cols)
            for l in range(cvs.shape[2]):
                if np.isnan(cvs[i,j,l,:].any(axis=0)):
                    continue
                out[i,j,l,0] = coords[i,j,1]#x-map
                mod[i,j,l,0] = coords[i,j,1]
                out[i,j,l,1] = coords[i,j,0]#y-map
                mod[i,j,l,1] = coords[i,j,0]
                
                temp = [i for i in cvs[i,j,l,:]]# cvs[i,j,l,0],cvs[i,j,l,1],cvs[i,j,l,2],cvs[i,j,l,3]                

                params, cov = curve_fit(Azimuth, x, temp)

                residuals = temp - Azimuth(x, *params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((temp-np.mean(temp))**2)
                if ss_tot == 0:
                    vars[1] = 1
                    break
                out[i,j,l,6] = 1-(ss_res/ss_tot)
                mod[i,j,l,2] = 1-(ss_res/ss_tot)

                Az0 = calc_angles(params[0],params[1])
                Abs0 = Azimuth(Az0, *params)
                Az1 = calc_angles(params[0],params[1])+90
                Abs1 = Azimuth(Az1, *params)
                Az2 = calc_angles(params[0],params[1])-90

                if vars[0] < 54.73:
                    if Abs0 > Abs1:
                        out[i,j,l,2] = Az0
                    elif Abs1 > Abs0:                        
                        if Az1 < 90:
                            out[i,j,l,2] = Az1
                        elif Az1 > 90:
                            out[i,j,l,2] = Az2
                elif vars[0] >= 54.73:
                    if Abs0 < Abs1:
                        out[i,j,l,2] = Az0
                    elif Abs1 < Abs0:                        
                        if Az1 < 90:
                            out[i,j,l,2] = Az1
                        elif Az1 > 90:
                            out[i,j,l,2] = Az2    

                out[i,j,l,3] = OrFunc(vars[0], *params)
                out[i,j,l,4]  = ampl1(*params)
                out[i,j,l,5]  = ampl2(params[0],params[1])            
                mod[i,j,l,3]  = params[0]
                mod[i,j,l,4]  = params[1]
                mod[i,j,l,5]  = params[2]

    tcvs.close()
    tout.close()
    tmod.close()
    tcoords.close()
    tvars.close()

def process_polar_abs(images, alpha, feature, map_x, map_y, invert, polangles, state):
    start = time.time()

    state.set_status("Preparing...")
    featnames = [i.name for i in feature]
    lsxs = np.empty(0)
    lsys = np.empty(0)
    for i in range(len(images)): 
        tempdata = images[i].transform(Domain([map_x, map_y]))
        lsx = np.unique(tempdata.X[:,0])
        lsy = np.unique(tempdata.X[:,1])
        lsxs = np.append(lsxs, lsx)
        lsys = np.append(lsys, lsy)

    ulsxs = np.unique(lsxs)
    ulsys = np.unique(lsys)

    cvs = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], len(featnames), len(images)), np.nan)
    spec = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], images[0].X.shape[1], len(images)), np.nan, dtype=object)
    metas = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], images[0].metas.shape[1], len(images)), np.nan, dtype=object)
    out = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], len(featnames), 7), np.nan)
    mod = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], len(featnames), 6), np.nan)
    coords = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], 2), np.nan)
    vars = np.asarray([alpha, 0])
    fill = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0]), np.nan)
    for i in range(len(images)): 
        cv = [images[i].domain[j] for j in featnames] #when feature is not meta type, have to recreate ContinuousVariable using name of original CV for some reason, otherwise returns NaNs for all except last image      
        doms = [map_x, map_y] + cv
        tempdata = images[i].transform(Domain(doms))      
        computevalue = pd.DataFrame(fill, index=ulsys, columns=ulsxs, dtype=object)
        attributes = pd.DataFrame(fill, index=ulsys, columns=ulsxs, dtype=object)
        meta = pd.DataFrame(fill, index=ulsys, columns=ulsxs, dtype=object)
        #####################################################
        #this is a time consuming loop (~2.27s for 36,864 spectra) i.e. with 4 loops(images/angles), ~82% of time to prepare data for calculations!!!
        for j, k in enumerate(tempdata):
            computevalue.at[k[1], k[0]] = k.x[2:]
            attributes.at[k[1], k[0]] = images[i].X[j]
            meta.at[k[1], k[0]] = images[i].metas[j]
        #####################################################
        # cvs[:,:,i] = computevalue.to_numpy(copy=True)
        for l, m in enumerate(attributes.columns):
            for n, o in enumerate(attributes.index):                
                cvs[n,l,:,i] = computevalue.at[o,m]
                spec[n,l,:,i] = attributes.at[o,m]
                metas[n,l,:,i] = meta.at[o,m]
    xys = pd.DataFrame(fill, index=ulsys, columns=ulsxs, dtype=object)
    for k, i in enumerate(xys.index):
        for l, j in enumerate(xys.columns):
            coords[k,l,0] = i
            coords[k,l,1] = j

    tcvs = shared_memory.SharedMemory(name='cvs', create=True, size=cvs.nbytes)
    scvs = np.ndarray(cvs.shape, dtype=cvs.dtype, buffer=tcvs.buf)
    scvs[:,:,:] = cvs[:,:,:]
    tout = shared_memory.SharedMemory(name='out', create=True, size=out.nbytes)
    sout = np.ndarray(out.shape, dtype=out.dtype, buffer=tout.buf)
    sout[:,:,:,:] = out[:,:,:,:]
    tmod = shared_memory.SharedMemory(name='mod', create=True, size=mod.nbytes)
    smod = np.ndarray(mod.shape, dtype=mod.dtype, buffer=tmod.buf)
    smod[:,:,:,:] = mod[:,:,:,:]
    tcoords = shared_memory.SharedMemory(name='coords', create=True, size=coords.nbytes)
    scoords = np.ndarray(coords.shape, dtype=coords.dtype, buffer=tcoords.buf)
    scoords[:,:,:] = coords[:,:,:]
    tvars = shared_memory.SharedMemory(name='vars', create=True, size=vars.nbytes)
    svars = np.ndarray(vars.shape, dtype=vars.dtype, buffer=tvars.buf)
    svars[:] = vars[:]

    shapes = [cvs.shape, spec.shape, metas.shape, out.shape, mod.shape, coords.shape, vars.shape]
    dtypes = [cvs.dtype, spec.dtype, metas.dtype, out.dtype, mod.dtype, coords.dtype, vars.dtype]

    ncpu = os.cpu_count()   
    # ncpu = 1
    tulsys = np.array_split(ulsys, ncpu)
    state.set_status("Calculating...")
    threads=[]
    cumu = 0
    for i in range(ncpu):
        tlsxys = [ulsxs,tulsys[i]]
        yidx = [cumu, cumu+len(tulsys[i])]
        cumu += len(tulsys[i])
        # compute(tlsxys, yidx, shapes, dtypes, polangles)
        t = multiprocessing.Process(target=compute, args=(tlsxys, yidx, shapes, dtypes, polangles))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    state.set_status("Finishing...")
    if invert == True:
        sout[:,:,:,2] = sout[:,:,:,2]*-1 
    outputs = np.reshape(sout[:,:,:,2:], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], 5*len(featnames)))
    model = np.reshape(smod[:,:,:,2:], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], 4*len(featnames)))
    
    spectra = []
    met = []
    for i in range(len(polangles)):
        spectratemp = np.reshape(spec[:,:,:,i], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].X.shape[1]))
        spectratemp = spectratemp[~np.isnan(model).any(axis=1)]
        spectra.append(spectratemp)
        metatemp = np.reshape(metas[:,:,:,i], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].metas.shape[1]))
        metatemp = metatemp[~np.isnan(model).any(axis=1)]
        metatemp = np.append(metatemp, np.full((np.shape(metatemp)[0],1), i), axis=1)
        met.append(metatemp)

    outputs = outputs[~np.isnan(model).any(axis=1)]    
    model = model[~np.isnan(model).any(axis=1)]

    spectra = np.concatenate((spectra), axis=0) 
    meta = np.concatenate((met), axis=0)

    tcvs.unlink()
    tout.unlink()
    tmod.unlink()
    tcoords.unlink()
    tvars.unlink()
    
    return outputs, model, spectra, meta, vars[1]

#calculate by "Stoke's Method"
# def compute_stokes(xys, yidx, shapes, dtypes):

#     tcvs = shared_memory.SharedMemory(name='cvs', create=False)
#     cvs = np.ndarray(shapes[0], dtype=dtypes[0], buffer=tcvs.buf)
#     tout = shared_memory.SharedMemory(name='out', create=False)
#     out = np.ndarray(shapes[3], dtype=dtypes[3], buffer=tout.buf)  
#     tcoords = shared_memory.SharedMemory(name='coords', create=False)
#     coords = np.ndarray(shapes[4], dtype=dtypes[4], buffer=tcoords.buf)

#     for i in range(yidx[0], yidx[1]):#y-values(rows)
#         for j, k in enumerate(xys[0]):#x-values(cols)
#             for l in range(cvs.shape[2]):
#                 if np.isnan(cvs[i,j,l,0]) == True or np.isnan(cvs[i,j,l,1]) == True or np.isnan(cvs[i,j,l,2]) == True or np.isnan(cvs[i,j,l,3]) == True:
#                     continue
#                 out[i,j,l,0] = coords[i,j,1]#x-map
#                 out[i,j,l,1] = coords[i,j,0]#y-map

#                 temp = [cvs[i,j,l,0],cvs[i,j,l,1],cvs[i,j,l,2],cvs[i,j,l,3]]

#                 out[i,j,l,2] = compute_theta(temp)
#                 out[i,j,l,3] = compute_amp(temp)
#                 out[i,j,l,4] = compute_intensity(temp)

#     tcvs.close()
#     tout.close()
#     tcoords.close()

# #Does not agree with other algorithm/published/reference data unless angles inverted
# def compute_theta(images):
#     return 0.5 * np.arctan2(images[1] - images[3], images[0] - images[2])


# def compute_intensity(images):
#     S0 = (images[0] + images[1] + images[2] + images[3]) * 0.5
#     return S0


# def compute_amp(images):
#     return np.sqrt((images[3] - images[1])**2 + (images[2] - images[0])**2) / compute_intensity(images)


# def process_polar_stokes(images, feature, map_x, map_y, invert, state):

#     state.set_status("Preparing...")
#     featnames = [i.name for i in feature]
#     lsxs = np.empty(0)
#     lsys = np.empty(0)
#     for i in range(len(images)): 
#         tempdata = images[i].transform(Domain([map_x, map_y]))
#         lsx = np.unique(tempdata.X[:,0])
#         lsy = np.unique(tempdata.X[:,1])
#         lsxs = np.append(lsxs, lsx)
#         lsys = np.append(lsys, lsy)

#     ulsxs = np.unique(lsxs)
#     ulsys = np.unique(lsys)

#     cvs = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], len(featnames), len(images)), np.nan)
#     spec = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], images[0].X.shape[1], len(images)), np.nan)
#     metas = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], images[0].metas.shape[1], len(images)), np.nan)
#     out = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], len(featnames), 5), np.nan)
#     coords = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0], 2), np.nan)
#     fill = np.full((np.shape(ulsys)[0], np.shape(ulsxs)[0]), np.nan)
#     for i in range(len(images)): 
#         cv = [images[i].domain[j] for j in featnames] #when feature is not meta type, have to recreate ContinuousVariable using name of original CV for some reason, otherwise returns NaNs for all except last image      
#         doms = [map_x, map_y] + cv
#         tempdata = images[i].transform(Domain(doms))      
#         computevalue = pd.DataFrame(fill, index=ulsys, columns=ulsxs, dtype=object)
#         attributes = pd.DataFrame(fill, index=ulsys, columns=ulsxs, dtype=object)
#         meta = pd.DataFrame(fill, index=ulsys, columns=ulsxs, dtype=object)
#         #####################################################
#         #this is a time consuming loop (~2.27s for 36,864 spectra) i.e. with 4 loops(images/angles), ~82% of time to prepare data for calculations!!!
#         for j, k in enumerate(tempdata):
#             computevalue.at[k[1], k[0]] = k.x[2:]
#             attributes.at[k[1], k[0]] = images[i].X[j]
#             meta.at[k[1], k[0]] = images[i].metas[j]
#         #####################################################

#         # cvs[:,:,i] = computevalue.to_numpy(copy=True)
#         for l, m in enumerate(attributes.columns):
#             for n, o in enumerate(attributes.index):
#                 cvs[n,l,:,i] = computevalue.at[o,m]
#                 spec[n,l,:,i] = attributes.at[o,m]
#                 metas[n,l,:,i] = meta.at[o,m]
#     xys = pd.DataFrame(fill, index=ulsys, columns=ulsxs, dtype=object)
#     for k, i in enumerate(xys.index):
#         for l, j in enumerate(xys.columns):
#             coords[k,l,0] = i
#             coords[k,l,1] = j

#     tcvs = shared_memory.SharedMemory(name='cvs', create=True, size=cvs.nbytes)
#     scvs = np.ndarray(cvs.shape, dtype=cvs.dtype, buffer=tcvs.buf)
#     scvs[:,:,:] = cvs[:,:,:]
#     tout = shared_memory.SharedMemory(name='out', create=True, size=out.nbytes)
#     sout = np.ndarray(out.shape, dtype=out.dtype, buffer=tout.buf)
#     sout[:,:,:,:] = out[:,:,:,:]
#     tcoords = shared_memory.SharedMemory(name='coords', create=True, size=coords.nbytes)
#     scoords = np.ndarray(coords.shape, dtype=coords.dtype, buffer=tcoords.buf)
#     scoords[:,:,:] = coords[:,:,:]


#     shapes = [cvs.shape, spec.shape, metas.shape, out.shape, coords.shape]
#     dtypes = [cvs.dtype, spec.dtype, metas.dtype, out.dtype, coords.dtype]

#     ncpu = os.cpu_count()     
#     tulsys = np.array_split(ulsys, ncpu)
#     state.set_status("Calculating...")
#     threads=[]
#     cumu = 0

#     for i in range(ncpu):
#         tlsxys = [ulsxs,tulsys[i]]
#         yidx = [cumu, cumu+len(tulsys[i])]
#         cumu += len(tulsys[i])
#         t = multiprocessing.Process(target=compute_stokes, args=(tlsxys, yidx, shapes, dtypes))
#         threads.append(t)
#         t.start()

#     for t in threads:
#         t.join()

#     state.set_status("Finishing...")
#     if invert == True:
#         sout[:,:,:,2] = sout[:,:,:,2]*-1 
#     outputs = np.reshape(sout[:,:,:,2:], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], 3*len(featnames)))

#     spectra0 = np.reshape(spec[:,:,:,0], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].X.shape[1]))
#     spectra45 = np.reshape(spec[:,:,:,1], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].X.shape[1]))
#     spectra90 = np.reshape(spec[:,:,:,2], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].X.shape[1]))
#     spectra135 = np.reshape(spec[:,:,:,3], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].X.shape[1]))
    
#     meta0 = np.reshape(metas[:,:,:,0], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].metas.shape[1]))
#     meta45 = np.reshape(metas[:,:,:,1], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].metas.shape[1]))
#     meta90 = np.reshape(metas[:,:,:,2], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].metas.shape[1]))
#     meta135 = np.reshape(metas[:,:,:,3], (np.shape(ulsys)[0]*np.shape(ulsxs)[0], images[0].metas.shape[1]))

       
#     spectra0 = spectra0[~np.isnan(outputs).any(axis=1)]
#     spectra45 = spectra45[~np.isnan(outputs).any(axis=1)]
#     spectra90 = spectra90[~np.isnan(outputs).any(axis=1)]
#     spectra135 = spectra135[~np.isnan(outputs).any(axis=1)]
#     meta0 = meta0[~np.isnan(outputs).any(axis=1)]
#     meta0 = np.append(meta0, np.full((np.shape(meta0)[0],1), 0), axis=1)
#     meta45 = meta45[~np.isnan(outputs).any(axis=1)]
#     meta45 = np.append(meta45, np.full((np.shape(meta45)[0],1), 1), axis=1)
#     meta90 = meta90[~np.isnan(outputs).any(axis=1)]
#     meta90 = np.append(meta90, np.full((np.shape(meta90)[0],1), 2), axis=1)
#     meta135 = meta135[~np.isnan(outputs).any(axis=1)]
#     meta135 = np.append(meta135, np.full((np.shape(meta135)[0],1), 3), axis=1)
#     outputs = outputs[~np.isnan(outputs).any(axis=1)] 

#     spectra = np.concatenate((spectra0, spectra45, spectra90, spectra135), axis=0) 
#     meta = np.concatenate((meta0, meta45, meta90, meta135), axis=0)

#     tcvs.unlink()
#     tout.unlink()
#     tcoords.unlink()

#     return outputs, spectra, meta


def hypercube_to_table(hc, wns, lsx, lsy):
    table = build_spec_table(*_spectra_from_image(hc,
                             wns,
                             np.linspace(*lsx),
                             np.linspace(*lsy)))
    return table

class OWPolar(OWWidget, ConcurrentWidgetMixin):
    
    # Widget's name as displayed in the canvas
    name = "4-Angle Polarisation 2"
    
    # Short widget description
    description = (
        "4-Angle Polarisation implimentation")

    icon = "icons/unknown.svg"    
    
    # Define inputs and outputs
    class Inputs:
        data = MultiInput("Data", Orange.data.Table, default=True)        

    class Outputs:
        polar = Output("Polar Data", Orange.data.Table, default=True)
        model = Output("Curve Fit model data",Orange.data.Table)

    autocommit = settings.Setting(False)

    settingsHandler = DomainContextHandler()

    want_main_area = False
    resizing_enabled = True
    alpha = ContextSetting(0)

    feature = ContextSetting(None)
    map_x = ContextSetting(None)
    map_y = ContextSetting(None)
    # method = Setting(0)
    invert_angles = Setting(False)
    
    angles = None
    anglst = Setting([], packable=False)
    lines = Setting([], packable=False)
    labels = Setting([], packable=False)
    multiin_anglst = Setting([], packable=False)
    multiin_lines = Setting([], packable=False)
    multiin_labels = Setting([], packable=False)
    minangles = 4
    polangles = Setting([], packable=False)
    n_inputs = 0

    feats: List[Variable] = Setting([])

    # method_names = ('Curve Fitting Method','Stokes Method')

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

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        gui.OWComponent.__init__(self)
        
        self._data_inputs: List[Optional[Table]] = []
        self.feats = None

        hbox = gui.hBox(self.controlArea, "4-Angle Polarisation")
        #col 1
                    
        self.vbox2 = gui.vBox(hbox, "Inputs")
        
        self.multifile = gui.widgetBox(self.vbox2, "Multifile Input (all angles in 1 table)")
        
        self.anglemetas = DomainModel(DomainModel.METAS, valid_types=DiscreteVariable)
        self.anglesel = gui.comboBox(self.multifile, self, 'angles', searchable=True, label='Select Angles by:', callback=self._change_angles, model=self.anglemetas)
        self.anglesel.setDisabled(True)
        
        self.multiin = gui.widgetBox(self.vbox2, "Multiple Inputs (1 angle per input)")   
                
        #col 2
        self.vbox1 = gui.vBox(hbox, "Features")
        # vbox1.setFixedSize()

        self.featureselect = DomainModel(DomainModel.SEPARATED,
            valid_types=ContinuousVariable)
        self.feat_view = ListViewSearch(selectionMode=QListView.ExtendedSelection)
        self.feat_view.setModel(self.featureselect)
        self.feat_view.selectionModel().selectionChanged.connect(self._feat_changed)
        self.vbox1.layout().addWidget(self.feat_view)
        self.vbox1.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum))
        
        #col 3
        vbox = gui.vBox(hbox, "Parameters")

        # mbox = gui.widgetBox(vbox, "Calculation method")

        # buttons = gui.radioButtons(
        #     mbox, self, "method",
        #     callback=self._change_input)

        # for opts in self.method_names:
        #     gui.appendRadioButton(buttons, self.tr(opts))

        form = QWidget()
        formlayout = QFormLayout()
        form.setLayout(formlayout)
        # mbox.layout().addWidget(form)

        # splitter = QSplitter(self)
        # splitter.setOrientation(Qt.Horizontal)        

        xybox = gui.widgetBox(vbox, "Data XY Selection")

        self.x_axis = DomainModel(DomainModel.METAS, valid_types=DomainModel.PRIMITIVE)
        self.y_axis = DomainModel(DomainModel.METAS, valid_types=DomainModel.PRIMITIVE)

        self.xvar = gui.comboBox(xybox, self, 'map_x', searchable=True, label="X Axis",
            callback=self._change_input, model=self.x_axis)
        self.yvar = gui.comboBox(xybox, self, 'map_y', searchable=True, label="Y Axis",
            callback=self._change_input, model=self.y_axis)

        vbox.layout().addWidget(form)
        gui.rubber(self.controlArea)

        self.alphavalue = gui.lineEdit(vbox, self, "alpha", "Alpha value", callback=self._change_input, valueType=int)

        gui.checkBox(vbox, self, 'invert_angles', label="Invert Angles", callback=self._change_input)#callback?

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply", commit=self.commit)
        self._change_input()
        self.contextAboutToBeOpened.connect(lambda x: self.init_attr_values(x[0]))
        self.resize(640, 300)


    def _feat_changed(self):
        rows = self.feat_view.selectionModel().selectedRows()
        values = self.feat_view.model()[:]
        self.feats = [values[row.row()] for row in sorted(rows)]
        self.commit.deferred()
        # self.commit.now()

    def init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.featureselect.set_domain(domain)
        self.x_axis.set_domain(domain)
        self.y_axis.set_domain(domain)
        self.anglemetas.set_domain(domain)
        self.group_x = None
        self.group_y = None

    def _change_input(self):
        # if self.method == 0:
        #     self.alphavalue.setDisabled(False)
        # elif self.method == 1:
        #     self.alphavalue.setDisabled(True)
        # self.commit.now()
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
                # self.commit.now() 
                self.commit.deferred() 
                              
    def add_angles(self, anglst, lab, labels, lines, widget, i, place, callback): #to be used in a loop   
        anglst.append(lab)
        ledit = gui.lineEdit(widget, self, None, label = lab, callback = callback)
        ledit.setText(str(place))
        lines.append(ledit)
        for j in ledit.parent().children():
            if type(j) is QLabel:
                labels.append(j)                        
    
    def clear_angles(self, anglst, lines, labels, widget):        
        for i in reversed(range(self.multiin.layout().count())): 
            self.multiin.layout().itemAt(i).widget().setParent(None)
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
            # self.commit.now()
            self.commit.deferred()
        except:
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
            # self.commit.now()
            self.commit.deferred()
        except:
            pass
    
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
        self.data = None 
        self.feats = None       
        self.closeContext()
        self.Warning.clear()
        self.Outputs.polar.send(None)
        self.Outputs.model.send(None)
        self.data = self.more_data
        self.clear_angles(self.anglst, self.lines, self.labels, self.multifile)
        self.clear_angles(self.multiin_anglst, self.multiin_lines, self.multiin_labels, self.multiin)
        
        node = self.signalManager.active_nodes()[0]
        inputlinks = self.signalManager.workflow().input_links(node)
        names = [name.source_node.title for name in inputlinks] 
                      
        tempangles = np.linspace(0, 180, len(self.data)+1) 
        for i in range(len(self.data)):            
            self.add_angles(self.multiin_anglst, names[i], self.multiin_labels, self.multiin_lines, 
                            self.multiin, i, tempangles[i], self._send_ind_angles)
            
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
        if len(self.data) == 0:
            self.Outputs.polar.send(None)
            self.Outputs.model.send(None)
            self.contextAboutToBeOpened.emit([Table.from_domain(Domain(()))])
            return
        
        if len(self.data) == 1:
            self.openContext(self.data[0])          
        elif 1 < len(self.data) < 4 or len(self.data) == 0:
            self.Warning.notenough()
            self.contextAboutToBeOpened.emit([Table.from_domain(Domain(()))])
            return        
        else:
            self.sorted_data = self.data            
            metas = []
            attrs = []
            class_vars = []
            for i in self.sorted_data:
                metas = metas+[j for j in i.domain.metas]
                attrs = attrs+[j for j in i.domain.attributes]
                class_vars = class_vars+[j for j in i.domain.class_vars]

            attrs = unify_tables(attrs)
            metas = unify_tables(metas)
            class_vars = unify_tables(class_vars)

            dom = Domain(attrs, metas = metas)

            for i, j in enumerate(self.sorted_data):
                self.sorted_data[i] = j.transform(dom)
            self.openContext(self.sorted_data[0])
            
        # self.commit.now()
        self.commit.deferred()

    @gui.deferred
    def commit(self):           
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
            if type(i) is not float:
                self.Warning.pol()
                return
        self.Warning.wrongdata.clear()
        
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
        
        
        
        # if self.method == 0:
        self.start(run0, sorted_data, self.feats, self.alpha, self.map_x, self.map_y, self.invert_angles, self.polangles) #, self.anglst, self.angles
        # elif self.method == 1:
        #     self.start(run1, sorted_data, self.feats, self.map_x, self.map_y, self.invert_angles)
           
    def on_done(self, result: Results):
        if result is None:
            self.Outputs.polar.send(None)
            self.Outputs.model.send(None)
            return
        if result.errorstate == 1:
            self.Warning.wrongdata()
        else:
            self.Outputs.polar.send(result.out)
            self.Outputs.model.send(result.model)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


        

if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPolar).run(Orange.data.Table("ftir-4pol.pkl.gz"))
