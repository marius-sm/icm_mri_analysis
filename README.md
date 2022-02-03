# ICM MRI analysis

This repo contains a pipeline that segments different structures on a T1 image, and then maps the segmentation masks to other images of the same subject (e.g. ASL) in order to measure the signal from these different structures.

## Usage

```bash
python run_pipeline.py -t1 T1.nii.gz -s .../ASL/ -st icm_choroid_plexus,lateral_ventricle,cortical_wm
```

Arguments:
- `-t1` Required. Should specify a path to the T1 image used for segmentation
- `-s` (or `--signal`) should specify a path to either one MRI volume, or a folder containing multiple volumes
- `-st` (or `--structures`) should be a list of structures which will be segmented, separated by a comma (but no spaces !). A list of the possible structures is available below.

## Available structures

`icm_choroid_plexus` will use a deep learning model trained on the dataset described in 
***
**Axial multi-layer perceptron architecture for automatic segmentation of choroid plexus in multiple sclerosis**  
Marius Schmidt-Mengin, Vito A. G. Ricigliano, Benedetta Bodini, Emanuele Morena, Annalisa Colombi, Mariem Hamzaoui, Arya Yazdan Panah, Bruno Stankoff, Olivier Colliot  
SPIE Medical Imaging 2022
***

All other structures will use FastSurfer
***
**FastSurfer - A fast and accurate deep learning based neuroimaging pipeline**  
Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M
NeuroImage 219 (2020), 117012
https://doi.org/10.1016/j.neuroimage.2020.117012
***

Here is a list of all possible structures

### Subcortical structures

`brain_stem`  
`third_ventricle`  
`fourth_ventricle`  
`csf`  
`wm_hypointensities`  
`cortical_wm` `left_cortical_wm` `right_cortical_wm`  
`lateral_ventricle` `left_lateral_ventricle` `right_lateral_ventricle`  
`inferior_lateral_ventricle` `left_inferior_lateral_ventricle` `right_inferior_lateral_ventricle`  
`cerebellar_wm` `left_cerebellar_wm` `right_cerebellar_wm`  
`cerebellar_cortex` `left_cerebellar_cortex` `right_cerebellar_cortex`  
`thalamus` `left_thalamus` `right_thalamus`  
`caudate` `left_caudate` `right_caudate`  
`putamen` `left_putamen` `right_putamen`  
`pallidum` `left_pallidum` `right_pallidum`  
`hippocampus` `left_hippocampus` `right_hippocampus`  
`amygdala` `left_amygdala` `right_amygdala`  
`accumbens` `left_accumbens` `right_accumbens`  
`ventral_dc` `left_ventral_dc` `right_ventral_dc`  
`choroid_plexus` `left_choroid_plexus` `right_choroid_plexus`  

### Cortical structures

`gm` `left_gm` `right_gm`  
`caudalanteriorcingulate` `left_caudalanteriorcingulate` `right_caudalanteriorcingulate`  
`caudalmiddlefrontal` `left_caudalmiddlefrontal` `right_caudalmiddlefrontal`  
`cuneus` `left_cuneus` `right_cuneus`  
`entorhinal` `left_entorhinal` `right_entorhinal`  
`fusiform` `left_fusiform` `right_fusiform`  
`inferiorparietal` `left_inferiorparietal` `right_inferiorparietal`  
`isthmuscingulate` `left_isthmuscingulate` `right_isthmuscingulate`  
`lateraloccipital` `left_lateraloccipital` `right_lateraloccipital`  
`lateralorbitofrontal` `left_lateralorbitofrontal` `right_lateralorbitofrontal`  
`lingual` `left_lingual` `right_lingual`  
`medialorbitofrontal` `left_medialorbitofrontal` `right_medialorbitofrontal`  
`middletemporal` `left_middletemporal` `right_middletemporal`  
`parahippocampal` `left_parahippocampal` `right_parahippocampal`  
`paracentral` `left_paracentral` `right_paracentral`  
`parsopercularis` `left_parsopercularis` `right_parsopercularis`  
`parsorbitalis` `left_parsorbitalis` `right_parsorbitalis`  
`parstriangularis` `left_parstriangularis` `right_parstriangularis`  
`pericalcarine` `left_pericalcarine` `right_pericalcarine`  
`postcentral` `left_postcentral` `right_postcentral`  
`posteriorcingulate` `left_posteriorcingulate` `right_posteriorcingulate`  
`precentral` `left_precentral` `right_precentral`  
`precuneus` `left_precuneus` `right_precuneus`  
`rostralanteriorcingulate` `left_rostralanteriorcingulate` `right_rostralanteriorcingulate`  
`rostralmiddlefrontal` `left_rostralmiddlefrontal` `right_rostralmiddlefrontal`  
`superiorfrontal` `left_superiorfrontal` `right_superiorfrontal`  
`superiorparietal` `left_superiorparietal` `right_superiorparietal`  
`superiortemporal` `left_superiortemporal` `right_superiortemporal`  
`supramarginal` `left_supramarginal` `right_supramarginal`  
`transversetemporal` `left_transversetemporal` `right_transversetemporal`  
`insula` `left_insula` `right_insula`  
