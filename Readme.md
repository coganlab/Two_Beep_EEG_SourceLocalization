# I RECOMEND YOU USE JUPYTER NOTEBOOK

LINUX SETUP:
1. Install freesurfer and mne onto your ubuntu
	a. make sure that you change your freesurfer environment variables to the folders you want them to be (on the shared drive between the linux distrib and the PC)
		Example:
			export FREESURFER_HOME=/share/freesurfer 
			export SUBJECTS_DIR=$FREESURFER_HOME/subjects 
			source $FREESURFER_HOME/SetUpFreeSurfer.sh
		Recomended - Select Shared or Box Folder

2. Install Xserver follow this website: _______
3. install anaconda-navigator or conda onto your linux distrib
4. Activate mne (conda activat mne)
5. open the gui for the navigator 
6. switch to the MNE driver
7. open spyder and a jupyter qt console
8. open a second Linux distrib to do coregistration
9. copy and paste all code from document into jupyter qt console
10. type %matplotlib qt --> allows for graphs to display
11. get environment variables so mne can work with yout FreeSurfer data
	a. FREESURFER_HOME = os.getenv('FREESURFER_HOME')
	b. SUBJECTS_DIR = os.getenv('SUBJECTS_DIR')
12. for 3D plotting do the following:
	a. mne.viz.set_3d_backend('pyvista')
	b. Export MNE_3D_OPTION_ANTIALIAS=false OR  mne.viz.set_3d_options(antialias=False)
		*Turns antialiasing*

'E15','E16','E18','E20','E25','E35' - best

PIPELINE:

1. Given your specific subject, create 2 new folders in their sub folder:
	Digitization - used for Localization
	Epochs - used for making the BEM

CLEAN DATA: *check out make_montage and change what files you traverse to based on your user and file structure*

	1. find Montage file (captrak) and load in EEG data and align it to correct channels
    montage, raw = make_montage()

	2. Common Average Reference and select noisy channels
    raw2 = set_data(raw, montage) #common average reference

	3. Save Captrak file to raw EEG file and then save it as a .fif
    raw2.save(pathlib.Path('/mnt/d/Aidan/' + patient + '/Digitization') / 'Captrak_Digitization.fif', overwrite = True)

	4. Check to see if you chose the right channels
    epochs = erp(raw2, montage)

	5. Run autorejection algorithms to detect noisy correlated epochs
    epochs_ar = autorej(epochs)

	6. Run an ICA on the autorejection data
    ica = ICA_auto_rej(epochs_ar)
    ica.plot_sources(raw2, show_scrollbars=True)     # to plot ICA vs time


	7. Remove ICA channels that correspond to artifacts
    ica.exclude = [0,1,2,5] # MANUAL INPUT

	8. Identify channels correlated with EOG data and remove them
    reconst_evoked, reconst_raw = EOG_check_ar(epochs_ar, ica)

	9. save cleaned file as evoked data to correct folder
    mne.write_evokeds('/mnt/d/Aidan/Grand_Averages/Epochs/' + subj + '_raw_for_ave.fif', reconst_evoked) #to do co-registration, have to do file path so gui.coreg can take as input

	10. save raw file
    cd grand_averages
    cd epochs
    reconst_raw.save(subj + '_reconst_raw.fif')
    cd ..
    cd ..


Coregistration:
	*to create a epochs_for_source_epo.fif file you first need to run covariance()*
	1. in qtconsole generate a bem
		epoch_data = '/mnt/d/'INSERT USERNAME'/'INSERT PATIENT OR SUBJECT'/Epochs/epochs_for_source_epo.fif'
			i.e. epoch_data = '/mnt/d/Aidan/E46/Epochs/epochs_for_source_epo.fif'

		epoch_info = mne.io.read_info(epoch_data)

		mne.bem.make_watershed_bem(subject=subj, subjects_dir=SUBJECTS_DIR, overwrite=True, volume='T1')
		
	2. Display bem
		mne.viz.plot_bem(subject = subj, subjects_dir = SUBJECTS_DIR, orientation = 'coronal')
		
	3. Double Check your file has fiducials
		mne.coreg.get_mni_fiducials(subj, SUBJECTS_DIR)
		
	4.Now open a new Ubuntu command line and activate mne
    		conda activate mne

	5. Perform Coregistration
    		type mne coreg into your Linux distribution
		make sure your digitization bem files are the same
		then click where your fiducials should be
		then hit fit fiducials

	6. save the Coregistration data as a trans.fif file
		hit the button on the coregistration gui (bottom right)

Localization: (you can do this half of the pipeline with epoched or evoked data, I used evoked)

	1. generate Noise covariance and baseline evoked data
    		noise_cov, fig_cov, fig_spectra, evoked = covariance(reconst_raw)
			- you can read in the raw:
				reconst_raw = mne.io.read_raw_fif('/mnt/d/Aidan/Grand_Averages/Epochs/' + subj + '_reconst_raw.fif')

	2. double check your patient is your current subject: i.e. E46
		subj

	3. define a path to traverse to your subjects given digitization info
		data_path = '/mnt/d/Aidan/' + subj + '/Digitization'

	4. save your digitization data to a specific variable
		data = pathlib.Path(data_path) / 'Captrak_Digitization.fif'

	5. read in that data from the file
		a. use baseline evoked
			info = mne.io.read_info(data)
		b. or use the evoked data you have already generated
			info = reconst_evoked.info
		c. or read in cleaned evoked
			reconst_evoked = mne.read_evokeds('/mnt/d/Aidan/Grand_Averages/Epochs/' + subj + '_raw_for_ave.fif')
			info = reconst_evoked[0].info
			
	6. read in your coregistration data
		trans = pathlib.Path(data_path)/ (subj + '-trans.fif')
		a. if your subjects MRI wasn't done correctly and you used the fsaverage brain
			trans = pathlib.Path(data_path)/ ('fsaverage-trans.fif')

	7. plot your coregistration data just to double check
		fig = mne.viz.plot_alignment(info=info, trans=trans, subject=patient, subjects_dir=SUBJECTS_DIR, dig=True, verbose=True)

	8. compute your space (2D) - oct4, oct5 or oct6 are recommended (these determine the number of sources), you can also change the 'conductance' (then save to the correct folder of your choice) 
		src = mne.setup_source_space(subject=subj, spacing='oct5', subjects_dir = SUBJECTS_DIR)
		cd Patient_SRC
		src.save(subj + '-src.fif',overwrite = True)
		cd ..		
		a. or read it in
			src= mne.read_source_spaces('/mnt/d/Aidan/Patient_SRC/' + subj+ '-src.fif')

	9. you can now display this alignment as well with your coregistration
		mne.viz.plot_alignment(info=info, trans=trans, subject=subj, src=src, subjects_dir=SUBJECTS_DIR, dig=True)

	10. Generate a BEM model
		model = mne.make_bem_model(subject=subj, subjects_dir=SUBJECTS_DIR)
		a. if you use the fsaverage brain
			model = mne.make_bem_model(subject='fsaverage', subjects_dir=SUBJECTS_DIR)

	11. Create a BEM 'solution'
		bem_sol=mne.make_bem_solution(model)
			*note* this takes a while
		a. or load it in
			bem_sol = mne.read_bem_solution('/mnt/d/Aidan/' + subj+ '/Digitization/' + subj+ '_bem.fif')

	12. Create a forward solution(mindist = ignore elcetrodes minimum distance from skull, n_jobs =  of jobs to run in parallel when computing, ignore_ref=True  can be used to ignore reference electrode (Common Average Reference))
		fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem_sol, meg=False, eeg=True, n_jobs=1) # mindist=5.0,
	
	13. save your bem, and fwd solutions
		bem_fname = pathlib.Path(data_path)/ 'E46_bem.fif'
		mne.bem.write_bem_solution(bem_fname, bem_sol, overwrite = True)
		fwd_fname = pathlib.Path(data_path)/'E46_fwd_soln.fif'
		mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

	14. Create inverse operator (optional variables: loose=?, depth=?)
		inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov)
			- Loose = the spread of the signal across the brain (horizantal neurons --> not pyramidal)
			- depth (only signif. if for not eloreta)

	15. determine SNR so you can set LAMBDA when applying inverse
		mne.viz.plot_snr_estimate(reconst_evoked, inv)
		snr, snr_est = estimate_snr(reconst_evoked,inv)

	16. Now apply the inverse operator to gain your Inverse Solution, (recommended method = 'sLORETA' or 'eLORETA')
		*NOTE*
			- set lambda = 1/SNR^2
		stc = mne.minimum_norm.apply_inverse(reconst_evoked, inv, method = 'eLORETA', lambda2 = 0.1111111111111111111111111111111)
		a. if you loaded in your reconst_evoked
			stc = mne.minimum_norm.apply_inverse(reconst_evoked[0], inv, method = 'eLORETA')

	
	17. plotting source space EEG SNR overtime
		snr_stc = stc.estimate_snr(reconst_evoked.info, fwd, noise_cov)
		snr_stc.plot(hemi = 'both',smoothing_steps = 5, surface = 'pial')
		------
		
		snr_stc = stc.estimate_snr(reconst_evoked[0].info, fwd, noise_cov)
		ave = np.mean(snr_stc.data, axis=0)

		plt.pyplot.figure()
		plt.pyplot.plot(reconst_evoked.times, ave)
		#ax.plot(reconst_evoked[0].times, ave)
		#ax.set(xlabel='Time (sec)', ylabel='SNR MEG-EEG')
		#fig.tight_layout()

Plotting:
	1. Plot the inverse solution
		brain = stc.plot()
	2. Use this variable to fine tune your plots
		surface_kwargs = dict(hemi = 'both', surface='inflated', subjects_dir = SUBJECTS_DIR, clim=dict(kind='value', lims=[3.5e-11, 6e-11, 8e-11]), views='lateral', initial_time=0.19, time_unit='s', size=(800,800), smoothing_steps=5)
			'surface' --> determines what parts of the brain will be viewed ('white', 'pial', 'inflated')
			'lims' -->  dictionary of bounds of amplitude activation
			'hemi' --> which hemisphere are you viewing

	3. replot
		brain = stc.plot(**surface_kwargs)
		stc.plot(hemi = 'split',smoothing_steps = 5, surface = 'pial')  # hemi = 'both'
	4. save
		cd *insert subject file name*
		stc.save('base_stc')
		cd ..

	

--------------------- NEW SOL - Fsaverage MORPH -------------------

1. Load in the src, stc and fsa_src
	#current_src = mne.read_source_spaces('/mnt/d/Aidan/Patient_src/' + subj + '-src.fif')
	fsa_src = mne.read_source_spaces(SUBJECTS_DIR + '/fsaverage/bem/fsaverage-ico-5-src.fif')
	#stc_path ='/mnt/d/Aidan/' + subj + '/base_stc-lh.stc'
	#stc_current = mne.source_estimate.read_source_estimate(stc_path)

2. Morph the stc to fsa space
	#indepth	- stc_morph_fsa = mne.compute_source_morph(current_src, subject_from=subj, subject_to='fsaverage', subjects_dir=SUBJECTS_DIR, zooms='auto', niter_affine=(100, 100, 10), niter_sdr=(5, 5, 3), spacing=5, smooth=None, warn=True, xhemi=False, sparse=False, src_to=fsa_src, precompute=False, verbose=False)
	#fwdsoln	- stc_morph_fsa = mne.compute_source_morph(current_src, subject_from=subj, subject_to='fsaverage', subjects_dir=SUBJECTS_DIR, src_to=fwd[fsa_src])
	#best		- stc_morph_fsa = mne.compute_source_morph(src, subject_from=subj, subject_to='fsaverage', subjects_dir=SUBJECTS_DIR, src_to=fsa_src)

3. apply the stc  
	stc_final = stc_morph_fsa.apply(stc) #stc_current

4. plot the morphed stcf
	stc_final.plot(hemi = 'split',smoothing_steps = 5, surface = 'pial')

5. Save the morphed STC
	cd Grand_averages
	cd Source_estimates
	stc_final.save(subj+ '_stc')


--- If you decide to read in data individually, re-do the pipeline ---
- read bem solution
	bem_sol = mne.read_bem_solution('/mnt/d/Aidan/' + subj+ '/Digitization/' + subj+ '_bem.fif')
- read reconst_evoked
	reconst_evoked = mne.read_evoked
	info = reconst_evoked.info
- read trans
	trans = mne.read_trans('/mnt/d/Aidan/' + patient + '/Digitization/' + subj+ '-trans.fif')

fwd_morph = mne.make_forward_solution(info, trans=trans, src=src_morph_fsa, bem=bem_sol, meg=False, eeg=True, mindist=5.0, n_jobs=1)
inv_morph = mne.minimum_norm.make_inverse_operator(info, fwd_morph, noise_cov)
stc_morph = mne.minimum_norm.apply_inverse(reconst_evoked, inv_morph, method = 'eLORETA')


## -- getting the ISI and jitter ----

time_events = mne.events_from_annotations(raw)
dif = 0
max_dif = 0
min_dif = 0
dif_total = 0
holder = 0
prev = 0
average_dif = 0
current = 0
for i in range (1,201):
    if (time_events[0])[i][2] < 3:
        if prev == 0:
            prev = time_events[0][i][0]
            holder = 1
        elif i == 200:
            current = time_events[0][i][0]
            dif = current-prev
            dif_total = dif + dif_total
            average_dif = dif_total/i        
        elif holder == 1:
            if dif == 0:
                current = time_events[0][i][0]
                dif = current-prev
                dif_total = dif + dif_total
                max_dif = dif
                min_dif = dif
                prev = current
                print('banana')
            else:
                current = time_events[0][i][0]
                dif = current-prev
                dif_total = dif + dif_total
                prev = current
                if dif > max_dif:
                    if dif > max_dif*2:
                        print(i)
                    else:
                        max_dif = dif
                elif dif < min_dif:
                    min_dif = dif
print(max_dif)
print(min_dif)
print(average_dif)
            