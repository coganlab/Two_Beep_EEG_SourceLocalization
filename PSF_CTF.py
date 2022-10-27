# SIMULATE SOURCE ACTIVITY - FINAL

# RUN FULL PIPELINE THEN

import os.path as op

import numpy as np

import mne
from mne.datasets import sample

from mne.minimum_norm import read_inverse_operator, apply_inverse
from mne.simulation import simulate_stc, simulate_evoked

labels = mne.read_labels_from_annot(subj, subjects_dir=SUBJECTS_DIR)
label_names = [label.name for label in labels]
n_labels = len(labels)

nave = 200
T = 150
times = np.linspace(-0.5, 1, T)
dt = times[1] - times[0]

seed = 42
lambda2 = 0.1111
signal = np.zeros((n_labels, T))
idx_lh = label_names.index('superiortemporal-lh')
signal[idx_lh, :] = 1e-7 * np.sin(7 * 2 * np.pi * times)
idx_rh = label_names.index('superiortemporal-rh')
signal[idx_rh, :] = 1e-7 * np.sin(7 * 2 * np.pi * times)

hemi_to_ind = {'lh': 0, 'rh': 1}
for i, label in enumerate(labels):
    # The `center_of_mass` function needs labels to have values.
    labels[i].values.fill(1.)

    # Restrict the eligible vertices to be those on the surface under
    # consideration and within the label.
    surf_vertices = fwd['src'][hemi_to_ind[label.hemi]]['vertno']
    restrict_verts = np.intersect1d(surf_vertices, label.vertices)
    com = labels[i].center_of_mass(subjects_dir=SUBJECTS_DIR,
                                   restrict_vertices=restrict_verts,
                                   surf='white')

    # Convert the center of vertex index from surface vertex list to Label's
    # vertex list.
    cent_idx = np.where(label.vertices == com)[0][0]
    if i  == idx_lh:
        cent_vertice_lh = label.vertices[cent_idx]
        cent_idx_lh = np.where(surf_vertices == cent_vertice_lh)[0][0]
        #cent_idx_lh_temp = cent_idx
        #cent_idx_lh = surf_vertices[tmp] 
    if i == idx_rh:
        cent_vertice_rh = label.vertices[cent_idx]
        cent_idx_rh = np.where(surf_vertices == cent_vertice_rh)[0][0]
        #cent_idx_rh_temp = cent_idx
        #print(label)
        #print(label.vertices[cent_idx_rh])
        #print(cent_idx)
        #print(cent_vertice_rh)
        #print(cent_idx_rh)
        #print(surf_vertices[cent_idx_rh])
        #cent_idx_rh = surf_vertices[tmp] 
    # Create a mask with 1 at center vertex and zeros elsewhere.
    labels[i].values.fill(0.)
    labels[i].values[cent_idx] = 1.


##-- attach signals to labels at centroid

stc_gen = simulate_stc(fwd['src'], labels, signal, times[0], dt,
                       value_fun=lambda x: x)

##-- plot

kwargs = dict(subjects_dir=SUBJECTS_DIR, hemi='split', smoothing_steps=5,
              time_unit='s', initial_time=0.05, size=1200,
              views=['lat', 'med'])
clim = dict(kind='value', pos_lims=[1e-9, 1e-8, 1e-7])
brain_gen = stc_gen.plot(clim=clim, **kwargs)

##-- get correct covariance - set epochs (NOT RECONST EVOKED - to have no bads so covariance works)
epochs.info['bads'] = []
cov = mne.compute_covariance(epochs, tmin=None, tmax=0.)

#cov = covariance_orig()
#*MAKE SURE YOU KNOW WHAT YOUR SUBJECT # IS*


--** ISSUE WITH INVERSE OPERATOR** --> 
mne.convert_forward_solution(fwd, surf_ori=True,
                             force_fixed=True, copy=False)

inv = mne.minimum_norm.make_inverse_operator(
    info=info, forward=fwd, noise_cov=noise_cov, loose=0.,
    depth=None)

inv_path = '/mnt/d/Aidan/Grand_Averages/Inv_op/'
mne.minimum_norm.write_inverse_operator(fname = inv_path + subj +'_inv.fif',inv=inv)
# this one reloaded
inv_reloaded = mne.minimum_norm.read_inverse_operator(inv_path +  subj +'_inv.fif')

##--simulate source space

evoked_gen = simulate_evoked(fwd, stc_gen, info, cov, nave,
                             random_state=seed)

# Map the simulated sensor-space data to source-space using the inverse
# operator.
stc_inv = apply_inverse(evoked_gen, inv_reloaded, lambda2, method='eLORETA')
stc_inv.plot(surface = 'inflated',hemi = 'split', smoothing_steps = 5)



--resolution MATRIX:

from mne.minimum_norm import (make_inverse_resolution_matrix, get_cross_talk,
                              get_point_spread)

rm_lor = make_inverse_resolution_matrix(fwd, inv_reloaded,
                                        method='eLORETA', lambda2=lambda2)

# get PSF and CTF for sLORETA at one vertex
#cent_idx_rh_idx = [cent_idx_rh]
#cent_idx_lh_idx = [cent_idx_lh]

#cent_idx_lh = [labels[61]]
#cent_idx_rh = [labels[62]]

cent_idx_rh_idx = [cent_idx_rh]
cent_idx_lh_idx = [cent_idx_lh]

stc_psf_lh = get_point_spread(rm_lor, fwd['src'], cent_idx_lh_idx, norm=True)

stc_ctf_lh = get_cross_talk(rm_lor, fwd['src'], cent_idx_lh_idx, norm=True)

stc_psf_rh = get_point_spread(rm_lor, fwd['src'], cent_idx_rh_idx, norm=True)

stc_ctf_rh = get_cross_talk(rm_lor, fwd['src'], cent_idx_rh_idx, norm=True)

#----- psf setup -----

# Which vertex corresponds to selected source
vertno_lh = fwd['src'][0]['vertno']
verttrue_lh = [vertno_lh[cent_idx_lh_idx[0]]]  # just one vertex

# find vertices with maxima in PSF and CTF
vert_max_psf_lh = vertno_lh[stc_psf_lh.data.argmax()]
vert_max_ctf_lh = vertno_lh[stc_ctf_lh.data.argmax()]


# Which vertex corresponds to selected source
vertno_rh = fwd['src'][1]['vertno']
verttrue_rh = [vertno_rh[cent_idx_rh_idx[0]]]  # just one vertex

# find vertices with maxima in PSF and CTF
vert_max_psf_rh = vertno_rh[stc_psf_rh.data.argmax()]
if stc_ctf_rh.data.argmax() > 1026:
	ta = stc_ctf_rh.data
	ta = ta[0:1026]
	vert_max_ctf_rh = vertno_rh[ta.argmax()]
else:
	vert_max_ctf_rh = vertno_rh[stc_ctf_rh.data.argmax()]


# -----LH - PSF -----

brain_psf = stc_psf_lh.plot(subj, 'inflated', 'lh', subjects_dir=SUBJECTS_DIR)
brain_psf.show_view('ventral')
brain_psf.add_text(0.1, 0.9, 'sLORETA PSF lh', 'title', font_size=16)

# True source location for PSF LEFT HEMI
brain_psf.add_foci(verttrue_lh, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Maximum of PSF LEFT HEMI
brain_psf.add_foci(vert_max_psf_lh, coords_as_verts=True, scale_factor=1.,
                   hemi='lh', color='black')

# -----RH - PSF -----

brain_psf_rh = stc_psf_rh.plot(subj, 'inflated', 'rh', subjects_dir=SUBJECTS_DIR)
brain_psf_rh.show_view('ventral')
brain_psf_rh.add_text(0.1, 0.9, 'sLORETA PSF rh', 'title', font_size=16)

# True source location for PSF
brain_psf_rh.add_foci(verttrue_rh, coords_as_verts=True, scale_factor=1., hemi='rh',
                   color='green')

# Maximum of PSF 
brain_psf_rh.add_foci(vert_max_psf_rh, coords_as_verts=True, scale_factor=1.,
                   hemi='rh', color='black')

#---- ctf ------

brain_ctf = stc_ctf_lh.plot(subj, 'inflated', 'lh', subjects_dir=SUBJECTS_DIR)
brain_ctf.add_text(0.1, 0.9, 'sLORETA CTF lh', 'title', font_size=16)
brain_ctf.show_view('ventral')

brain_ctf_rh = stc_ctf_rh.plot(subj, 'inflated', 'rh', subjects_dir=SUBJECTS_DIR)
brain_ctf_rh.add_text(0.1, 0.9, 'sLORETA CTF rh', 'title', font_size=16)
brain_ctf_rh.show_view('ventral')

# -----LH - CTF -----
brain_ctf.add_foci(verttrue_lh, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Maximum of CTF
brain_ctf.add_foci(vert_max_ctf_lh, coords_as_verts=True, scale_factor=1.,
                   hemi='lh', color='black')

# -----RH - CTF -----
brain_ctf_rh.add_foci(verttrue_rh, coords_as_verts=True, scale_factor=1., hemi='rh',
                   color='green')

# Maximum of CTF - RH
brain_ctf_rh.add_foci(vert_max_ctf_rh, coords_as_verts=True, scale_factor=1.,
                   hemi='rh', color='black')


------------------------ COMPARING DIFFERENT METHODS -------------------

from mne.minimum_norm import make_inverse_resolution_matrix
from mne.minimum_norm import resolution_metrics

## ------------------------PSF-----------------------

# -- MNE --

rm_mne = make_inverse_resolution_matrix(fwd, inv_reloaded,
                                        method='MNE', lambda2=lambda2)
ple_mne_psf = resolution_metrics(rm_mne, inv_reloaded['src'],
                                 function='psf', metric='peak_err')
sd_mne_psf = resolution_metrics(rm_mne, inv_reloaded['src'],
                                function='psf', metric='sd_ext')

tmp_sd_M = sd_mne_psf.to_data_frame()
tmp_ple_M = ple_mne_psf.to_data_frame()
tmp_sd_M.to_csv('/mnt/d/Aidan/Grand_Averages/SD_PSF/sd_mne_psf_' + subj + '.csv')
tmp_ple_M.to_csv('/mnt/d/Aidan/Grand_Averages/PLE_PSF/ple_mne_psf_' + subj + '.csv')

# -- dSPM --

rm_dspm = make_inverse_resolution_matrix(fwd, inv_reloaded,
                                        method='dSPM', lambda2=lambda2)
ple_dspm_psf = resolution_metrics(rm_dspm, inv_reloaded['src'],
                                 function='psf', metric='peak_err')
sd_dspm_psf = resolution_metrics(rm_dspm, inv_reloaded['src'],
                                function='psf', metric='sd_ext')
tmp_sd_ds = sd_dspm_psf.to_data_frame()
tmp_ple_ds = ple_dspm_psf.to_data_frame()
tmp_sd_ds.to_csv('/mnt/d/Aidan/Grand_Averages/SD_PSF/sd_dspm_psf_' + subj + '.csv')
tmp_ple_ds.to_csv('/mnt/d/Aidan/Grand_Averages/PLE_PSF/ple_dspm_psf_' + subj + '.csv')

# -- sLORETA --

rm_sLORETA = make_inverse_resolution_matrix(fwd, inv_reloaded,
                                        method='sLORETA', lambda2=lambda2)
ple_sLORETA_psf = resolution_metrics(rm_sLORETA, inv_reloaded['src'],
                                 function='psf', metric='peak_err')
sd_sLORETA_psf = resolution_metrics(rm_sLORETA, inv_reloaded['src'],
                                function='psf', metric='sd_ext')

tmp_sd_s = sd_sLORETA_psf.to_data_frame()
tmp_ple_s = ple_sLORETA_psf.to_data_frame()
tmp_sd_s.to_csv('/mnt/d/Aidan/Grand_Averages/SD_PSF/sd_sLORETA_psf_' + subj + '.csv')
tmp_ple_s.to_csv('/mnt/d/Aidan/Grand_Averages/PLE_PSF/ple_sLORETA_psf_' + subj + '.csv')



#---------------- CTF -------------------

# -- MNE --

ple_mne_ctf = resolution_metrics(rm_mne, inv_reloaded['src'],
                                 function='ctf', metric='peak_err')
sd_mne_ctf = resolution_metrics(rm_mne, inv_reloaded['src'],
                                function='ctf', metric='sd_ext')

ctmp_sd_M = sd_mne_ctf.to_data_frame()
ctmp_ple_M = ple_mne_ctf.to_data_frame()
ctmp_sd_M.to_csv('/mnt/d/Aidan/Grand_Averages/SD_CTF/sd_mne_ctf_' + subj + '.csv')
ctmp_ple_M.to_csv('/mnt/d/Aidan/Grand_Averages/PLE_CTF/ple_mne_ctf_' + subj + '.csv')

# -- dSPM --

ple_dspm_ctf = resolution_metrics(rm_dspm, inv_reloaded['src'],
                                 function='ctf', metric='peak_err')
sd_dspm_ctf = resolution_metrics(rm_dspm, inv_reloaded['src'],
                                function='ctf', metric='sd_ext')

ctmp_sd_ds = sd_dspm_ctf.to_data_frame()
ctmp_ple_ds = ple_dspm_ctf.to_data_frame()
ctmp_sd_ds.to_csv('/mnt/d/Aidan/Grand_Averages/SD_CTF/sd_dspm_ctf_' + subj + '.csv')
ctmp_ple_ds.to_csv('/mnt/d/Aidan/Grand_Averages/PLE_CTF/ple_dspm_ctf_' + subj + '.csv')

# -- sLORETA --

ple_sLORETA_ctf = resolution_metrics(rm_sLORETA, inv_reloaded['src'],
                                 function='ctf', metric='peak_err')
sd_sLORETA_ctf = resolution_metrics(rm_sLORETA, inv_reloaded['src'],
                                function='ctf', metric='sd_ext')

ctmp_sd_s = sd_sLORETA_ctf.to_data_frame()
ctmp_ple_s = ple_sLORETA_ctf.to_data_frame()
ctmp_sd_s.to_csv('/mnt/d/Aidan/Grand_Averages/SD_CTF/sd_sLORETA_ctf_' + subj + '.csv')
ctmp_ple_s.to_csv('/mnt/d/Aidan/Grand_Averages/PLE_CTF/ple_sLORETA_ctf_' + subj + '.csv')



# -- VISUALIZING --


## -- MNE, dSPM, sLORETA PLE --
brain_ple_mne = ple_mne_psf.plot(subj, 'inflated', 'lh',
                                 subjects_dir=SUBJECTS_DIR, figure=1,
                                 clim=dict(kind='value', lims=(0, 2, 4)))
brain_ple_mne.add_text(0.1, 0.9, 'PLE MNE', 'title', font_size=16)

brain_ple_dspm = ple_dspm_psf.plot(subj, 'inflated', 'lh',
                                   subjects_dir=SUBJECTS_DIR, figure=2,
                                   clim=dict(kind='value', lims=(0, 2, 4)))
brain_ple_dspm.add_text(0.1, 0.9, 'PLE dSPM', 'title', font_size=16)

brain_ple_sLORETA = ple_sLORETA_psf.plot(subj, 'inflated', 'lh',
                                   subjects_dir=SUBJECTS_DIR, figure=3,
                                   clim=dict(kind='value', lims=(0, 2, 4)))
brain_ple_sLORETA.add_text(0.1, 0.9, 'PLE sLORETA', 'title', font_size=16)

## -- MNE, dSPM, sLORETA SD --

brain_sd_mne = sd_mne_psf.plot(subj, 'inflated', 'lh',
                               subjects_dir=SUBJECTS_DIR, figure=4,
                               clim=dict(kind='value', lims=(0, 2, 4)))
brain_sd_mne.add_text(0.1, 0.9, 'SD MNE', 'title', font_size=16)

brain_sd_dspm = sd_dspm_psf.plot(subj, 'inflated', 'lh',
                                 subjects_dir=SUBJECTS_DIR, figure=5,
                                 clim=dict(kind='value', lims=(0, 2, 4)))
brain_sd_dspm.add_text(0.1, 0.9, 'SD dSPM', 'title', font_size=16)

brain_sd_sLORETA = sd_sLORETA_psf.plot(subj, 'inflated', 'lh',
                                 subjects_dir=SUBJECTS_DIR, figure=6,
                                 clim=dict(kind='value', lims=(0, 2, 4)))
brain_sd_sLORETA.add_text(0.1, 0.9, 'SD sLORETA', 'title', font_size=16)

## -- MNE and dSPM difference for sd and PLE --

diff_ple = ple_mne_psf - ple_dspm_psf

brain_ple_diff = diff_ple.plot(subj, 'inflated', 'lh',
                               subjects_dir=SUBJECTS_DIR, figure=7,
                               clim=dict(kind='value', pos_lims=(0., 1., 2.)))
brain_ple_diff.add_text(0.1, 0.9, 'PLE MNE-dSPM', 'title', font_size=16)

diff_sd = sd_mne_psf - sd_dspm_psf

brain_sd_diff = diff_sd.plot(subj, 'inflated', 'lh',
                             subjects_dir=SUBJECTS_DIR, figure=8,
                             clim=dict(kind='value', pos_lims=(0., 1., 2.)))
brain_sd_diff.add_text(0.1, 0.9, 'SD MNE-dSPM', 'title', font_size=16)



#################################################################################
## GETTING PLE ----------------------


stc_psf_lh_mne = get_point_spread(rm_mne, fwd['src'], cent_idx_lh_idx, norm=True)

stc_ctf_lh_mne = get_cross_talk(rm_mne, fwd['src'], cent_idx_lh_idx, norm=True)

stc_psf_rh_mne = get_point_spread(rm_mne, fwd['src'], cent_idx_rh_idx, norm=True)

stc_ctf_rh_mne = get_cross_talk(rm_mne, fwd['src'], cent_idx_rh_idx, norm=True)

#--
stc_psf_lh_dspm = get_point_spread(rm_dspm, fwd['src'], cent_idx_lh_idx, norm=True)

stc_ctf_lh_dspm = get_cross_talk(rm_dspm, fwd['src'], cent_idx_lh_idx, norm=True)

stc_psf_rh_dspm = get_point_spread(rm_dspm, fwd['src'], cent_idx_rh_idx, norm=True)

stc_ctf_rh_dspm = get_cross_talk(rm_dspm, fwd['src'], cent_idx_rh_idx, norm=True)

#--
stc_psf_lh_sLORETA = get_point_spread(rm_sLORETA, fwd['src'], cent_idx_lh_idx, norm=True)

stc_ctf_lh_sLORETA = get_cross_talk(rm_sLORETA, fwd['src'], cent_idx_lh_idx, norm=True)

stc_psf_rh_sLORETA = get_point_spread(rm_sLORETA, fwd['src'], cent_idx_rh_idx, norm=True)

stc_ctf_rh_sLORETA = get_cross_talk(rm_sLORETA, fwd['src'], cent_idx_rh_idx, norm=True)

locations_lh = src[0]['rr'][vertno_lh, :]
locations_rh = src[1]['rr'][vertno_rh, :]
locations = np.vstack([locations_lh, locations_rh])


vert_max_psf_lh_mne = vertno_lh[stc_psf_lh_mne.data.argmax()]
vert_max_ctf_lh_mne = vertno_lh[stc_ctf_lh_mne.data.argmax()]

if stc_psf_rh_mne.data.argmax() > 1026:
	ta = stc_psf_rh_mne.data
	ta_psf_rmne = ta[0:1026]
	vert_max_psf_rh_mne = vertno_rh[ta_psf_rmne.argmax()]
else:
	ta_psf_rmne = stc_psf_rh_mne.data
	vert_max_psf_rh_mne = vertno_rh[ta_psf_rmne.argmax()]
if stc_ctf_rh_mne.data.argmax() > 1026:
	ta = stc_ctf_rh_mne.data
	ta_ctf_rmne = ta[0:1026]
	vert_max_ctf_rh_mne = vertno_rh[ta_psf_rmne.argmax()]
else:
	ta_ctf_rmne = stc_ctf_rh_mne.data
	vert_max_ctf_rh_mne = vertno_rh[ta_ctf_rmne.argmax()]



vert_max_psf_lh_dspm = vertno_lh[stc_psf_lh_dspm.data.argmax()]
vert_max_ctf_lh_dspm = vertno_lh[stc_ctf_lh_dspm.data.argmax()]


if stc_psf_rh_dspm.data.argmax() > 1026:
	ta_psf_rds = stc_psf_rh_dspm.data
	ta_psf_rds = ta_psf_rds[0:1026]
	vert_max_psf_rh_dspm = vertno_rh[ta_psf_rds.argmax()]
else:
	ta_psf_rds  = stc_psf_rh_dspm.data
	vert_max_psf_rh_dspm = vertno_rh[ta_psf_rds.argmax()]
if stc_ctf_rh_dspm.data.argmax() > 1026:
	ta = stc_ctf_rh_dspm.data
	ta_ctf_rds = ta[0:1026]
	vert_max_ctf_rh_dspm = vertno_rh[ta_ctf_rds.argmax()]
else:
	ta_ctf_rds = stc_ctf_rh_dspm.data
	vert_max_ctf_rh_dspm = vertno_rh[ta_ctf_rds.argmax()]


vert_max_psf_lh_sLORETA = vertno_lh[stc_psf_lh_sLORETA.data.argmax()]
vert_max_ctf_lh_sLORETA = vertno_lh[stc_ctf_lh_sLORETA.data.argmax()]


if stc_psf_rh_sLORETA.data.argmax() > 1026:
	ta = stc_psf_rh_sLORETA.data
	ta_psf_rsl = ta[0:1026]
	vert_max_psf_rh_sLORETA = vertno_rh[ta_psf_rsl.argmax()]
else:
	ta_psf_rsl = stc_psf_rh_sLORETA.data
	vert_max_psf_rh_sLORETA = vertno_rh[ta_psf_rsl.argmax()]
if stc_ctf_rh_sLORETA.data.argmax() > 1026:
	ta = stc_ctf_rh_sLORETA.data
	ta_ctf_rsl = ta[0:1026]
	vert_max_ctf_rh_sLORETA = vertno_rh[ta_ctf_rsl.argmax()]
else:
	ta_ctf_rsl = stc_ctf_rh_sLORETA.data
	vert_max_ctf_rh_sLORETA = vertno_rh[ta_ctf_rsl.argmax()]

locerr_total = []
#mne
true_loc_lh_mne_psf = locations_lh[cent_idx_lh,:]
vertmax_loc_lh_mne_psf = locations_lh[stc_psf_lh_mne.data.argmax(),:]
true_loc_rh_mne_psf = locations_rh[cent_idx_rh,:]
diffloc = true_loc_lh_mne_psf - vertmax_loc_lh_mne_psf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

vertmax_loc_rh_mne_psf = locations_rh[ta_psf_rmne.argmax(),:]
diffloc = true_loc_rh_mne_psf - vertmax_loc_rh_mne_psf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

true_loc_lh_mne_ctf = locations_lh[cent_idx_lh,:]
vertmax_loc_lh_mne_ctf = locations_lh[stc_ctf_lh_mne.data.argmax(), :]
diffloc = true_loc_lh_mne_ctf - vertmax_loc_lh_mne_ctf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

true_loc_rh_mne_ctf = locations_rh[cent_idx_rh,:]
vertmax_loc_rh_mne_ctf = locations_rh[ta_ctf_rmne.argmax(), :]
diffloc = true_loc_rh_mne_ctf - vertmax_loc_rh_mne_ctf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

#dspm
true_loc_lh_dspm_psf = locations_lh[cent_idx_lh,:]
vertmax_loc_lh_dspm_psf = locations_lh[stc_psf_lh_dspm.data.argmax(), :]
diffloc = true_loc_lh_dspm_psf - vertmax_loc_lh_dspm_psf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

true_loc_rh_dspm_psf = locations_rh[cent_idx_rh,:]
vertmax_loc_rh_dspm_psf = locations_rh[ta_psf_rds.argmax(), :]
diffloc = true_loc_rh_dspm_psf - vertmax_loc_rh_dspm_psf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

true_loc_lh_dspm_ctf = locations_lh[cent_idx_lh,:]
vertmax_loc_lh_dspm_ctf = locations_lh[stc_ctf_lh_dspm.data.argmax(), :]
diffloc = true_loc_lh_dspm_ctf - vertmax_loc_lh_dspm_ctf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

true_loc_rh_dspm_ctf = locations_rh[cent_idx_rh,:]
vertmax_loc_rh_dspm_ctf = locations_rh[ta_ctf_rds.argmax(), :]
diffloc = true_loc_rh_dspm_ctf - vertmax_loc_rh_dspm_ctf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)


#sLORETA

true_loc_lh_sLORETA_psf = locations_lh[cent_idx_lh,:]
vertmax_loc_lh_sLORETA_psf = locations_lh[stc_psf_lh_sLORETA.data.argmax(), :]
diffloc = true_loc_lh_sLORETA_psf - vertmax_loc_lh_sLORETA_psf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

true_loc_rh_sLORETA_psf = locations_rh[cent_idx_rh,:]
vertmax_loc_rh_sLORETA_psf = locations_rh[ta_psf_rsl.argmax(), :]
diffloc = true_loc_rh_sLORETA_psf - vertmax_loc_rh_sLORETA_psf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

true_loc_lh_sLORETA_ctf = locations_lh[cent_idx_lh,:]
vertmax_loc_lh_sLORETA_ctf = locations_lh[stc_ctf_lh_sLORETA.data.argmax().argmax(), :]
diffloc = true_loc_lh_sLORETA_ctf - vertmax_loc_lh_sLORETA_ctf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)

true_loc_rh_sLORETA_ctf = locations_rh[cent_idx_rh,:]
vertmax_loc_rh_sLORETA_ctf = locations_rh[ta_ctf_rsl.argmax(), :]
diffloc = true_loc_rh_sLORETA_ctf - vertmax_loc_rh_sLORETA_ctf    # diff btw true locs and maxima locs
locerr = np.linalg.norm(diffloc, axis=0)  # Euclidean distance
print(locerr)
locerr_total.append(locerr)


for i in range (len(locerr_total)):
    locerr_total[i] = locerr_total[i]*100
