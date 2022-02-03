import time
import os
import argparse
import glob
import sys
import subprocess
import torchio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import numpy as np
import re
from collections import OrderedDict
import pandas as pd

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
subcortical_structure_map = {
    'cortical_wm': 2,
    'lateral_ventricle': 4,
    'inferior_lateral_ventricle': 5,
    'cerebellar_wm': 7,
    'cerebellar_cortex': 8,
    'thalamus': 10,
    'caudate': 11,
    'putamen': 12,
    'pallidum': 13,
}

subcortical_structure_map = {
    **{k: [v, v+39] for k, v in subcortical_structure_map.items()},
    **{'left_'+k: [v] for k, v in subcortical_structure_map.items()},
    **{'right_'+k: [v+39] for k, v in subcortical_structure_map.items()}
}

subcortical_structure_map = {**subcortical_structure_map,
    'left_hippocampus': [17],
    'left_amygdala': [18],
    'left_accumbens': [26],
    'left_ventral_dc': [28],
    'left_choroid_plexus': [31],
                 
    'right_hippocampus': [53],
    'right_amygdala': [54],
    'right_accumbens': [58],
    'right_ventral_dc': [60],
    'right_choroid_plexus': [63], 
                 
    'hippocampus': [17, 53],
    'amygdala': [18, 54],
    'accumbens': [26, 58],
    'ventral_dc': [28, 60],
    'choroid_plexus': [31, 63] 
}

subcortical_structure_map = {
    'left_gm': [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 10],
    'brain_stem': [16],
    'third_ventricle': [14],
    'fourth_ventricle': [15],
    'csf': [24],
    'wm_hypointensities': [77],
    **subcortical_structure_map
}

cortical_structure_map = {
    'caudalanteriorcingulate': 1002,
    'caudalmiddlefrontal': 1003,
    'cuneus': 1005,
    'entorhinal': 1006,
    'fusiform': 1007,
    'inferiorparietal': 1008,
    'isthmuscingulate': 1010,
    'lateraloccipital': 1011,
    'lateralorbitofrontal': 1012,
    'lingual': 1013,
    'medialorbitofrontal': 1014,
    'middletemporal': 1015,
    'parahippocampal': 1016,
    'paracentral': 1017,
    'parsopercularis': 1018,
    'parsorbitalis': 1019,
    'parstriangularis': 1020,
    'pericalcarine': 1021,
    'postcentral': 1022,
    'posteriorcingulate': 1023,
    'precentral': 1024,
    'precuneus': 1025,
    'rostralanteriorcingulate': 1026,
    'rostralmiddlefrontal': 1027,
    'superiorfrontal': 1028,
    'superiorparietal': 1029,
    'superiortemporal': 1030,
    'supramarginal': 1031,
    'transversetemporal': 1034,
    'insula': 1035,
}

cortical_structure_map = {
    'gm': [v for v in cortical_structure_map.values()] + [v+1000 for v in cortical_structure_map.values()],
    'left_gm': [v for v in cortical_structure_map.values()],
    'right_gm': [v+1000 for v in cortical_structure_map.values()],
    **{k: [v, v+1000] for k, v in cortical_structure_map.items()},
    **{'left_'+k: [v] for k, v in cortical_structure_map.items()},
    **{'right_'+k: [v+1000] for k, v in cortical_structure_map.items()}
}

structures_map = {**subcortical_structure_map, **cortical_structure_map}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t1','--t1', help='(Required) Path to the T1 image used for segmentation.', required=True)
    parser.add_argument('-s','--signal', help='(Optional) Path to the image containing the signal to measure (ASL, CBF, T1 weighted perfusion...). This can also be a folder containing multiple images.', required=False)
    parser.add_argument('-r','--register', action='store_true', help='(Optional) Add this flag to register the CFB/ASL images onto the T1 image using FLIRT.', required=False)
    #parser.add_argument('-no_fs', '--no_fastsurfer', action='store_true', help='(Optional) Add this flag to bipass FastSurfer.', required=False)
    #parser.add_argument('-no_cp', '--no_plexus', action='store_true', help='(Optional) Add this flag to bipass plexus choroid segmentation.', required=False)
    parser.add_argument('-st', '--structures', help='(Optional) Add other structures to segment.', required=False, default='plexus, plexus')
    parser.add_argument('-fa','--flirt_args', help='(Optional) Additional arguments to pass to FLIRT, for example for changing the cost function.', required=False, default='')
    args = parser.parse_args()
        
    structures = args.structures.split(',')
    for s in structures:
        if s not in structures_map.keys():
            if s not in ['icm_choroid_plexus', 'all']:
                raise ValueError(f'{bcolors.FAIL}Unrecognized structure: {s}{bcolors.ENDC}')
                
    if 'icm_choroid_plexus' in structures:
        plexus_seg = True
    else:
        plexus_seg = False
    
    fs_structures = set(structures) - set(['icm_choroid_plexus', 'all'])
    if len(fs_structures) > 0:
        fastsurfer = True
    else:
        fastsurfer = False
    
    #print(structures)
    
    #sys.exit()
    
    #fs_label_map = {
    #    'White matter': [2, 41],
    #    'Grey matter': ['>1000'],
    #    'Thalami': [10, 49],
    #    'Ventricle': [4, 5, 43, 44]
    #}
    #fs_label_map = OrderedDict(fs_label_map)
    
    t1_file = os.path.abspath(args.t1)
    
########## Collect signal files ###############################################
    # look for images in the args.signal folder
    signal_files = [] # signal stands for quantity of interest.
    signal_abspath = os.path.abspath(args.signal)
    if not os.path.isdir(signal_abspath):
        signal_files = [signal_abspath]
    else:
        def block_print(): sys.stdout = open(os.devnull, 'w'); sys.stderr = open(os.devnull, 'w')
        def enable_print(): sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__
        for p in os.listdir(args.signal):
            file_path = os.path.join(signal_abspath, p)
            block_print()
            try:
                image = torchio.ScalarImage(file_path)
                image.load()
                signal_files.append(file_path)
            except Exception as e:
                pass
            enable_print()
        if len(signal_files) == 0:
            print(f'{bcolors.FAIL}WARNING: found no signal files. Segmentation will be run anyway.{bcolors.ENDC}')
        print(f'Found {len(signal_files)} valid signal files:')
        for f in signal_files:
            print(f' - {f}')
    signal_files = list(sorted(signal_files))
    
########## Create output dir ###############################################
    output_dir = os.path.join(os.path.dirname(t1_file), 'roi_pipeline_' + os.path.basename(t1_file))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        print('Creating', output_dir)

########## Choroid plexus segmentation ###############################################
    if plexus_seg:
        print('\n\n\n\nChoroid plexus segmentation ===========================\n')
        plexus_mask_file = os.path.join(output_dir, 'plexus_mask.nii.gz')
        if not os.path.isfile(plexus_mask_file):
            command = f'python plexus_segmentation.py -i {t1_file} -o {plexus_mask_file}'
            print(f'Running:\n{bcolors.OKCYAN}{command}{bcolors.ENDC}')
            subprocess.run(command, shell=True)
        else:
            print(f'Using the following pre-existing choroid plexus mask file: {plexus_mask_file}')
        if not os.path.isfile(plexus_mask_file):
            print(f'{bcolors.FAIL}ERROR: choroid plexus segmentation failed.{bcolors.ENDC}')
            raise Exception
    
########## FastSurfer segmentation ###############################################
    if fastsurfer:
        print('\n\n\n\nFastSurfer segmentation ==============================\n')
        fastsurfer_sid = 'fastsurfer'
        fastsurfer_sd = output_dir # This way, the outputs of FastSurfer will be saved in output_dir/fastsurfer
        fs_seg_file = os.path.join(fastsurfer_sd, fastsurfer_sid, 'mri/aparc.DKTatlas+aseg.deep.mgz')
        fs_seg_file_nifti = fs_seg_file[:-3] + 'nii.gz'
        def to_native_space(fs_seg_file, fs_seg_file_nifti):
            if os.path.isfile(fs_seg_file):
                command = f'mri_vol2vol --mov {fs_seg_file} --targ {t1_file} --o {fs_seg_file_nifti} --regheader --interp nearest'
                print(f'Aligning FastSurfer output file to back to native space:\n{bcolors.OKCYAN}{command}{bcolors.ENDC}')
                subprocess.run(command, shell=True)
            if not os.path.isfile(fs_seg_file_nifti):
                print(f'{bcolors.FAIL}ERROR: alignment to native space failed.{bcolors.ENDC}')
                raise Exception
        if os.path.isfile(fs_seg_file_nifti): 
            print(f'Using the following pre-existing segmentation file: {fs_seg_file_nifti}')
        elif os.path.isfile(fs_seg_file):
            print(f'Using the following pre-existing segmentation file: {fs_seg_file}')
            to_native_space(fs_seg_file, fs_seg_file_nifti)
        else:
            command = f'cd {os.path.join(os.path.abspath("."), "FastSurfer")}; ./run_fastsurfer.sh --t1 {t1_file} --sd {fastsurfer_sd} --sid {fastsurfer_sid} --seg_only'
            print(f'Running FastSurfer:\n{bcolors.OKCYAN}{command}{bcolors.ENDC}')
            subprocess.run(command, shell=True)
            if not os.path.isfile(fs_seg_file):
                print(f'{bcolors.FAIL}ERROR: FastSurfer segmentation failed.{bcolors.ENDC}')
                raise Exception
            else:
                to_native_space(fs_seg_file, fs_seg_file_nifti)
        fs_seg_file = fs_seg_file_nifti
    
########## Registration ###############################################    
    if args.register:
        print('\n\n\n\nRegistration of CBF onto T1 ==========================\n')
        
        registered_dir = os.path.join(output_dir, 'registered_CBF')
        
        if not os.path.isdir(registered_dir):
            os.mkdir(registered_dir)
        
        def get_registered_filename(file):
            output = os.path.join(registered_dir, 'registered_' + os.path.basename(file))
            if output[-3:] != '.gz':
                output = output + '.gz'
            return output
        
        flirt_commands = ''
        for f in signal_files:
            if os.path.isfile(get_registered_filename(f)):
                print(f'Using the following pre-existing registered file: {get_registered_filename(f)}')
            else:
                command = f'flirt -in {f} -ref {t1_file} -out {get_registered_filename(f)} -v {args.flirt_args}'
                print(f'{bcolors.OKCYAN}{command}{bcolors.ENDC}')
                flirt_commands += command + ' & '
                
        flirt_commands += 'wait'
        subprocess.run(flirt_commands, shell=True)
        
        for f in signal_files:
            if not os.path.isfile(get_registered_filename(f)):
                print(f'{bcolors.FAIL}ERROR: registration failed for {f}{bcolors.ENDC}')
                raise Exception

########## Alignment in T1 space ###############################################
# Actually, this does not seem to be necessary
#    else:
#        print('\n\n\n\nAlignment of signal onto T1 =============================\n')
#        
#        aligned_dir = os.path.join(output_dir, 'aligned_signal')
#        
#        if not os.path.isdir(aligned_dir):
#            os.mkdir(aligned_dir)
#        
#        def get_aligned_filename(file):
#            output = os.path.join(aligned_dir, 'aligned_' + os.path.basename(file))
#            return output
#        
#        align_commands = ''
#        for f in signal_files:
#            if os.path.isfile(get_aligned_filename(f)):
#                print(f'Using the following pre-existing aligned file: {get_aligned_filename(f)}')
#            else:
#                command = f'mri_vol2vol --mov {f} --targ {t1_file} --o {get_aligned_filename(f)} --regheader'
#                print(f'{bcolors.OKCYAN}{command}{bcolors.ENDC}')
#                align_commands += command + ' & '
#        align_commands += 'wait'
#        subprocess.run(align_commands, shell=True)
#        
#        for f in signal_files:
#            if not os.path.isfile(get_aligned_filename(f)):
#                print(f'{bcolors.FAIL}ERROR: alignment failed for {f}{bcolors.ENDC}')
#                raise Exception
        
########## Computation and visualization ###############################################
    print('\n\n\n\nGenerating visualization and curves =================\n')
    
    def threeview(tensor):
        # tensor must have shape (c, d, h, w)
        # the orientation is supposed to be RAS+
        c, d, h, w = tensor.shape
        images = [tensor[:, d//2, :, :], tensor[:, :, h//2, :], tensor[:, :, :, w//2]]
        images = [torch.rot90(i, k=1, dims=(1, 2)) for i in images]
        max_height = max([s.shape[1] for s in images])
        images = [F.pad(i, (0, 0, 0, max_height-i.shape[1])) for i in images]
        image = torch.cat(images, 2)
        return image
    
    
    t1 = torchio.ScalarImage(t1_file)
    t1 = torchio.transforms.Resample(1)(t1)
    t1 = torchio.transforms.RescaleIntensity((0, 1), percentiles=(0.5, 99.5))(t1)
    t1 = torchio.transforms.ToCanonical()(t1)
    t1_image = threeview(t1.tensor)
    t1_image_pil = Image.fromarray((t1_image.numpy()*255).astype(np.uint8)[0]).convert('RGBA')
    
    if plexus_seg:
        plexus_mask = torchio.ScalarImage(plexus_mask_file)
        plexus_mask = torchio.transforms.Resample(target=t1)(plexus_mask)
        #plexus_mask_fullres_tensor = plexus_mask.tensor.clone()[0]
        plexus_mask = torchio.transforms.ToCanonical()(plexus_mask)
        plexus_mask_image = threeview(plexus_mask.tensor)
        cm = plt.get_cmap('winter')
        plexus_mask_image_pil = cm(plexus_mask_image)
        plexus_mask_image_pil[..., -1] = plexus_mask_image
        plexus_mask_image_pil = Image.fromarray((plexus_mask_image_pil*255).astype(np.uint8)[0])
        
    if fastsurfer:
        fs_seg = torchio.ScalarImage(fs_seg_file)
        #fs_seg_fullres_tensor = fs_seg.tensor.clone()[0]
        #fs_seg_fullres_labels = np.zeros(fs_seg_fullres_tensor.shape, dtype=np.int32)
        fs_seg = torchio.transforms.Resample(target=t1, image_interpolation='nearest')(fs_seg)
        fs_seg = torchio.transforms.ToCanonical()(fs_seg)
        #fs_seg = torchio.transforms.Resample(1, image_interpolation='nearest')(fs_seg)
        fs_seg_image = threeview(fs_seg.tensor)[0]
        fs_seg_image_color = torch.zeros_like(fs_seg_image).float()
        
        for i, struc in enumerate(fs_structures):
            c = (float(i)+1)/len(fs_structures)
            labels = structures_map[struc]
            fs_seg_image_color[np.isin(fs_seg_image, labels)] = c
            #fs_seg_fullres_labels[np.isin(fs_seg_fullres_tensor.numpy(), labels)] = i
        
        #for i, (label, selection) in enumerate(fs_label_map.items()):
        #    x = (float(i)+1)/len(fs_label_map)
        #    selection_int = [s for s in selection if isinstance(s, int)]
        #    selection_str = [s for s in selection if isinstance(s, str)]
        #    if len(selection_int) > 0:
        #        fs_seg_image_color[np.isin(fs_seg_image, selection_int)] = x
        #        fs_seg_fullres_labels[np.isin(fs_seg_fullres_tensor.numpy(), selection_int)] = i
        #    for sstr in selection_str:
        #        fs_seg_image_color[eval(f'fs_seg_image {sstr}')] = x
        #        fs_seg_fullres_labels[eval(f'fs_seg_fullres_tensor.numpy() {sstr}')] = i
        
        cm = plt.get_cmap('plasma')
        alpha = (fs_seg_image_color > 0).float().numpy()
        fs_seg_image_color = cm(fs_seg_image_color)
        fs_seg_image_color[..., -1] = alpha
        fs_seg_image_pil = Image.fromarray((fs_seg_image_color*255).astype(np.uint8))
    
    if plexus_seg and not fastsurfer:
        seg_image_pil = fs_seg_image_pil
    if (not plexus_seg) and fastsurfer:
        seg_image_pil = plexus_mask_image_pil
    if plexus_seg and fastsurfer:
        seg_image_pil = Image.alpha_composite(fs_seg_image_pil, plexus_mask_image_pil)
    seg_image_pil = Image.alpha_composite(t1_image_pil, seg_image_pil)
    
    cm = plt.get_cmap('rainbow')
    final_image_stack = []
    means = [['Structure volume']]
    for j, sfile in enumerate(signal_files):
        print(f'Processing {os.path.basename(sfile)}...')
        aligned_sfile = get_registered_filename(sfile) if args.register else sfile
        signal = torchio.ScalarImage(aligned_sfile)
        signal = torchio.transforms.Resample(target=t1)(signal)
        signal = torchio.transforms.ToCanonical()(signal)
        
        means_row = [os.path.basename(sfile)]
        for struct in structures:
            if struct == 'all':
                means_row.append(signal.tensor.mean().item())
                if len(means) == 1:
                    means[0].append(signal.tensor.numel()* t1.spacing[0] * t1.spacing[1] * t1.spacing[2])
                continue
            elif struct == 'icm_choroid_plexus':
                mask = plexus_mask.tensor.numpy()
            else:
                idx = structures_map[struct]
                mask = np.zeros_like(fs_seg.tensor.numpy())
                mask[np.isin(fs_seg.tensor.numpy(), idx)] = 1
            vol = mask.sum() * t1.spacing[0] * t1.spacing[1] * t1.spacing[2]
            if len(means) == 1:
                means[0].append(vol)
            if vol == 0:
                means_row.append(0)
            else:
                mean = (signal.tensor.numpy() * mask).sum()/vol
                means_row.append(mean.item())
        means.append(means_row)
        
        signal = torchio.transforms.RescaleIntensity((0, 1), percentiles=(0.5, 99.5))(signal)
        
        signal_image = threeview(signal.tensor)[0]

        signal_image_colored = cm(signal_image)
        signal_image_colored[:, :, -1] = signal_image.numpy()
        signal_image_pil = Image.fromarray((255*signal_image_colored).astype(np.uint8))
        composite_image = Image.alpha_composite(t1_image_pil, signal_image_pil)
        
        composite_image = np.array(composite_image)
        composite_image = np.concatenate([np.zeros((30, composite_image.shape[1], composite_image.shape[2]), dtype=composite_image.dtype), composite_image], 0)
        composite_image = Image.fromarray(composite_image)
        
        draw = ImageDraw.Draw(composite_image)
        font = ImageFont.truetype('verdana.ttf', 16)
        draw.text((5, 5), f'{os.path.basename(sfile)}', (0,0,0), font=font)
        
        final_image_stack.append(np.array(composite_image))
        
    final_image_stack = [np.array(seg_image_pil)] + final_image_stack
    final_image_stack = np.concatenate(final_image_stack, 0)
    final_image_stack_pil = Image.fromarray(final_image_stack)
    final_image_stack_pil.save(os.path.join(output_dir, 'visualization.png'))
    
    df_means = pd.DataFrame(means, columns=['Input'] + structures)
    df_means.to_csv(os.path.join(output_dir, 'outputs.csv'), index=False)
    
    print('')
    print(df_means.to_string(index=False))
  