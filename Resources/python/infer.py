#!/usr/bin/env python3


import argparse
import math
import sys
import cv2
import os
import datetime
import logging, logging.handlers
import logging.config


#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import tensorflow as tf
import numpy as np


from fcnvgg import FCNVGG
from utils import *
from glob import glob
from tqdm import tqdm

class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height
#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate data based on a model')

parser.add_argument('--name', type=str, default='', #path to learnt net, end without \\
                    help='project name')

parser.add_argument('--checkpoint', type=int, default=-1,
                    help='checkpoint to restore; -1 is the most recent')

parser.add_argument('--batch-size', type=int, default=2,
                    help='batch size')

parser.add_argument('--data-source', default='cityscapes',
                    help='data source')
#nowe 
parser.add_argument('--input_dir', type=str, default='', # avis path need end on \\
                    help='Directory containing avi files for specific session and specific camera id. Source to analyse.')

parser.add_argument('--output_dir', type=str, default='',# destination path, need end on \\
                    help="Directory containing analysed avi files for specific session and specific camera id. Results.")

parser.add_argument('--images_ext', default='_image.avi', 
                    help="Extension to result images file name.")

parser.add_argument('--labels_ext', default='_label.avi', 
                    help="Extension to result labels file name.")

parser.add_argument('--frames_buffer_size', type=int, default=16,
                    help='Frames buffer size.')

parser.add_argument('--is_output_printing', type=bool, default=1, 
                    help = 'Is output data printing? (false - only log in file)')

parser.add_argument('--is_org_camera_saving', type=bool, default=1, 
                    help = 'Is oryginal camera session saving?')

parser.add_argument('--cuda_visible_devices', type=str, default='0',
                    help='Cuda visible devices')

args, unparsed = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices

global orgial_image_size
global target_image_size
global expanded_org_image_size
global expanded_target_image_size
global overlay_percent
global sizes_initialized

orgial_image_size = Size(height = 0, width = 0) #pobrane przy pobieraniu obrazkow
target_image_size = Size(height = 0, width = 0) #pobrane na wstepie z source.image_size
expanded_org_image_size = Size(height = 0, width = 0) #przeliczone zaraz po pobraniu orgial_image_size
expanded_target_image_size = Size(height = 0, width = 0) #rozszerzony docelowy rozmiar obrazu
overlay_percent = 0.05
sizes_initialized = 0

#------------------------------------------------------------------------------
#Logers Configuration
#------------------------------------------------------------------------------
def setup_logger(logger_name, log_file, level=logging.INFO):

    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)   
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)

def logger(msg, level, logfile):
 
    if logfile == 'log_simple':
        log = logging.getLogger('log_simple') 
    else:
        if logfile == 'log_output':
            log = logging.getLogger('log_output') 

    if level == 'info':
        log.info(msg) 
    else:
        if level == 'warning':
            log.warning(msg) 
        else:
            if level == 'error':
                log.error(msg)

logs_path = os.getenv('APPDATA')+"\\P.U.T. GRAW Sp. z o.o\\VI.AOD.AutomaticObjectDetection\\logs"

try:
    os.makedirs(logs_path, exist_ok=True)
except OSError:
    print("Creation of the directory "+ logs_path +" failed")

log_time_start = datetime.datetime.now().strftime("%d%m%Y_%H%M")
log_output_name =  logs_path + "\\infer_output_" + log_time_start +".log"
setup_logger("log_output", log_output_name)

log_simple_name = logs_path + "\\infer_logs_" + log_time_start +".log"
setup_logger("log_simple", log_simple_name)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def write_to_output(content, always_print, log_lvl):
    if always_print or args.is_output_printing:
        print(content)
        logger(content, log_lvl, 'log_output')        
     
    logger(content, log_lvl, 'log_simple')

#-------------------------------------------------------------------------------
def sample_generator(no_of_samples, position, image_size, batch_size, out_image, vidcap):

    all_images = []
    images = []
    names  = []

    file_is_loaded = False
    counter =0
    loop_counter = 0 

    while file_is_loaded == False:

        if int(no_of_samples) <= args.frames_buffer_size:

            for iterator in range(0, int(no_of_samples)):   
                if vidcap.grab() == False:
                    break
                success, source = vidcap.retrieve()
                if success:
                    all_images.append(source) 

            for offset in range(0, int(no_of_samples), batch_size):        
                yield prepare_samples(batch_size, all_images, image_size, offset)

            file_is_loaded = True

        else:   

            super_frame = 0
            if counter == 0:
               super_frame = 1 # przy pierwszej iteracji pobiera o jeden obrazk dalej, tak, żeby był o krok dalej

            for iterator in range(0, args.frames_buffer_size + super_frame):   
                if vidcap.grab() == False:
                    break
                success, source = vidcap.retrieve()
                if success:
                    all_images.append(source)
                    counter += 1

            end_offset =  (args.frames_buffer_size * (loop_counter +1)) 
            
            if end_offset > no_of_samples:            
                end_offset = no_of_samples            

            for offset in range(loop_counter * args.frames_buffer_size, int(end_offset) , batch_size): 
                yield prepare_samples(batch_size, all_images, image_size, offset)

            if counter >= no_of_samples: 
                file_is_loaded = True                                           
            
            loop_counter = loop_counter + 1
     
def initialize_sizes(org_img_height, org_ima_width):
    global orgial_image_size    
    global expanded_org_image_size
    global overlay_percent
    global sizes_initialized

    orgial_image_size = Size(height=org_img_height, width=org_ima_width)
    expanded_org_image_size = calculate_expended_image_size(orgial_image_size, overlay_percent)
    sizes_initialized = 1
          
def prepare_samples(batch_size, all_images, image_size, offset): 
    global orgial_image_size
    global expanded_target_image_size
    global target_image_size
    #---------------------------------------------------------------------------
    # Rewrite image sizes
    #---------------------------------------------------------------------------        

    if sizes_initialized == 0:
        initialize_sizes(org_img_height =  all_images[0].shape[0], org_ima_width = all_images[0].shape[1])

    imgs = all_images[offset:offset+batch_size]
    images = []
    names  = []
    for i in range(0, batch_size):                

        if i >= len(imgs):
                break

        if offset + i - 1 <= 0:      
            prev_img =  np.empty(shape=all_images[0].shape)
        else:
            prev_img =all_images[offset+i-1]

        if offset + i + 1  >= len(all_images):            
           next_image = np.empty(shape=all_images[0].shape)
        else:           
            next_image = all_images[offset+i+1] 
                
        exp_image = prepare_expanded_picture(prev_img, all_images[offset+i], next_image, orgial_image_size, overlay_percent) 
        resized_export_img = cv2.resize(exp_image,  (expanded_target_image_size.width,expanded_target_image_size.height))
        image_out =  cv2.resize(imgs[i], (target_image_size.width, target_image_size.height))     
        out_image.write(image_out)             
                       
        images.append(resized_export_img.astype(np.float32))
        names.append(str(position))
        
    return np.array(images), names


def crop_nump_image(image, image_size, height_start, height_stop):
   
    if height_start < 0:
        height_start = 0        
    
    if height_stop > image_size.height: 
        height_stop = image_size.height

    result_image = image[height_start:height_stop,:,:]
    return result_image

def calculate_expended_image_size(orgial_image_size, overlay_percent):
    overlay_height = int(orgial_image_size.height * overlay_percent)     
    exp_img_size = Size(height=orgial_image_size.height+2*overlay_height, width=orgial_image_size.width)
    return exp_img_size

def prepare_expanded_picture(prev_image, image, next_image, image_size, overlay_percent):
    image_height = image_size.height
    overlay_height = int(image_height * overlay_percent)     

    part1_start = image_height - overlay_height
    part1_end = image_height
    part2_start = 0
    part2_end = image_height
    part3_start = 0
    part3_end = overlay_height    

    part1 = crop_nump_image(prev_image, image_size, part1_start, part1_end)
    part2 = crop_nump_image(image, image_size, part2_start, part2_end)
    part3 = crop_nump_image(next_image, image_size, part3_start, part3_end)

    result_image = np.concatenate((part1, part2, part3),0)
    
    return result_image

def cut_image_to_target_size(image, target_img_size, expanded_img_size):
    cut_start = int((expanded_img_size.height-target_img_size.height)/2)
    cut_end = int(cut_start + target_img_size.height)
    result_image = crop_nump_image(image, expanded_img_size, cut_start, cut_end)
    return result_image


def calculate_target_expended_image_size(target_img_size, overlay_percent):
    expanded_height = int(target_img_size.height*(2*overlay_percent))
    expanded_height = expanded_height + ((32 - expanded_height % 32) % 32) 
    target_expended_img_size = Size(height = target_img_size.height+expanded_height, width = target_img_size.width)
    return target_expended_img_size

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
write_to_output('Project name:      ' + args.name, 0, 'info')
write_to_output('Output directory:  ' + args.output_dir, 0, 'info')
write_to_output('Samples directory: ' + args.input_dir, 0, 'info')
write_to_output('Batch size:        ' + str(args.batch_size), 0, 'info')

#-------------------------------------------------------------------------------
# Check if we can get the checkpoint
#-------------------------------------------------------------------------------
state = tf.train.get_checkpoint_state(args.name)
if state is None:   
    write_to_output('[!] No network state found in ' + args.name, 1, 'error')
    sys.exit(1)
try:
    checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]   
    write_to_output('[i] Network checkpoint:' + checkpoint_file, 0, 'info')    
except IndexError:
    write_to_output('[!] Cannot find checkpoint ' + str(args.checkpoint), 1, 'error')    
    sys.exit(1)

metagraph_file = checkpoint_file + '.meta'
write_to_output('[i] Metagraph file:    ' + metagraph_file, 0, 'info')

if not os.path.exists(metagraph_file):    
    write_to_output('[!] Cannot find metagraph ' + metagraph_file, 1, 'info')   
    sys.exit(1)

#-------------------------------------------------------------------------------
# Load the data source
#-------------------------------------------------------------------------------
try:
    source = load_data_source(args.data_source)   
    label_colors = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    write_to_output('[!] Unable to load data source: ' + str(e), 1, 'error')     
    sys.exit(1)

#-------------------------------------------------------------------------------
# Create a list of files to analyse and make sure that the output directory
# exists
#-------------------------------------------------------------------------------
# INPUT DIRECTORY VALIDATION
#-------------------------------------------------------------------------------
if os.path.exists(args.input_dir):
    input_dirs = glob(args.input_dir + '/*.avi')
    # Check if AVI exists:
    if len(input_dirs) ==  0: 
        write_to_output("The source directory does not contain any AVI files.", 1, 'error') 
        sys.exit(1)
else:
    write_to_output("The source directory does not exists.", 1, 'error')    
    sys.exit(1)
#-------------------------------------------------------------------------------
# OUTPUT DIRECTORY VALIDATION
#-------------------------------------------------------------------------------
if not os.path.exists(args.output_dir):
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        write_to_output('The output directory cannot be created: ' + str(e), 1, 'error')
        sys.exit(1)

#-------------------------------------------------------------------------------
# Check AVI files (output) and remove if exists
#-------------------------------------------------------------------------------
output_dirs = glob(args.output_dir + '/*.avi')  
for output_dir in output_dirs:
    try:
        os.remove(output_dir)
    except OSError as e:
        write_to_output('Cannot remove the file: ' + output_dir + str(e), 1, 'error')
        sys.exit(1)

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------

write_to_output('[i] Project name:      ' + args.name, 0, 'info')
write_to_output('[i] Network checkpoint:' + checkpoint_file, 0, 'info')
write_to_output('[i] Metagraph file:    ' + metagraph_file, 0, 'info')
write_to_output('[i] Output directory:  ' + args.output_dir, 0, 'info')
write_to_output('[i] Samples directory: ' + args.input_dir, 0, 'info')
write_to_output('[i] Image size:        ' + str(source.image_size), 0, 'info')
write_to_output('[i] # classes:         ' + str(source.num_classes), 0, 'info')
write_to_output('[i] Batch size:        ' + str(args.batch_size), 0, 'info')
#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
device_name = '/device:GPU:'+args.cuda_visible_devices
#device_name = "/cpu:0"
with tf.device(device_name):
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9    

    target_image_size = Size(width = source.image_size[0], height = source.image_size[1]) #zadana wiekosc obrazu, wynikowy format     
    expanded_target_image_size =  calculate_target_expended_image_size(target_image_size, overlay_percent) #obraz rozszerzony

    with tf.Session(config=config) as sess:
        write_to_output('Creating the model...', 0, 'info')
        net = FCNVGG(sess)
        net.build_from_metagraph(metagraph_file, checkpoint_file)
        print('Files numbers:         ' + str(len(input_dirs)))
        for input_dir in input_dirs:
            #---------------------------------------------------------------------------
            # Process the images
            #---------------------------------------------------------------------------
            write_to_output('Process the images...', 0, 'info')
            write_to_output('File: ' + input_dir, 0, 'info')
            position = 0

            file_name = os.path.splitext(os.path.basename(input_dir))[0]   

            output_images_file_name = ""
            if args.is_org_camera_saving:
                output_images_file_name = args.output_dir+file_name+args.images_ext
            output_labels_file_name = args.output_dir+file_name+args.labels_ext      
            #---------------------------------------------------------------------------
            # Open Video Writer
            #---------------------------------------------------------------------------        
            
            out_image = cv2.VideoWriter(output_images_file_name, cv2.VideoWriter_fourcc(*'HFYU'), 5, (target_image_size.width, target_image_size.height))
            out_label = cv2.VideoWriter(output_labels_file_name, cv2.VideoWriter_fourcc(*'HFYU'), 5, (target_image_size.width, target_image_size.height))
           
            #---------------------------------------------------------------------------
            # Open AVI file
            #---------------------------------------------------------------------------            
            vidcap = cv2.VideoCapture(input_dir) 
            
            print('Frames numbers:         ' +str(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))))
            if(vidcap.isOpened == False):
                write_to_output('Cannot open file: ' + input_dir, 1, 'error')
                break
            no_of_samples =vidcap.get(cv2.CAP_PROP_FRAME_COUNT)  
            n_sample_batches = int(math.ceil(no_of_samples/args.batch_size))
            description = '\n[i] Processing samples'  
            generator = sample_generator(no_of_samples, position, expanded_target_image_size, args.batch_size, out_image, vidcap)
            
            for x, names in tqdm(generator, total=n_sample_batches, desc=description, unit='batches'):
                feed = {net.image_input:  x,
                        net.keep_prob:    1}
                img_labels = sess.run(net.classes, feed_dict=feed)                
                expanded_labels = draw_labels_batch1(x, img_labels, label_colors, False) 

                for i in range(len(names)):
                    label = cut_image_to_target_size(expanded_labels[i], target_image_size, expanded_target_image_size) 
                    out_label.write(np.uint8(label))     
                    print('Frame processed')         

                position = position + args.batch_size
                
            #---------------------------------------------------------------------------
            # Close Video Writer
            #---------------------------------------------------------------------------          
            out_image.release()
            out_label.release()
            #---------------------------------------------------------------------------
            # Close AVI file
            #---------------------------------------------------------------------------
            vidcap.release()
            write_to_output('File ' +input_dir+ ' done.', 0, 'info')
            write_to_output('Output file with images: ' + output_images_file_name + ' done.', 0, 'info' )
            write_to_output('Output file with labels: ' + output_labels_file_name + ' done.', 0, 'info')    
write_to_output('[i] All done.', 0, 'info')

