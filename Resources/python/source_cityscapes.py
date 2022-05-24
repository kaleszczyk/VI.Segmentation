

#-------------------------------------------------------------------------------

import random
import cv2
import os

import numpy as np
import xml.etree.ElementTree as et

from collections import namedtuple
from glob import glob

#-------------------------------------------------------------------------------
# Labels
#-------------------------------------------------------------------------------

#def rgb2bgr(tpl):
#    return (tpl[2], tpl[1], tpl[0])

def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])

Label = namedtuple('Label', ['name', 'color'])
#-------------------------------------------------------------------------------
# Import labels colours from xml 
#-------------------------------------------------------------------------------
tree = et.parse(os.path.dirname(os.path.abspath(__file__)) + "\colours.xml")
p = tree.find("Labels")
names = list(tree.iter("Name"))
rs = list(tree.iter("R"))
gs = list(tree.iter("G"))
bs = list(tree.iter("B"))
label_defs = []

for i in range(0, len(rs)): 
    label = Label(names[i], rgb2bgr((int(rs[i].text), int(gs[i].text), int(bs[i].text))))
    label_defs.append(label)   


#-------------------------------------------------------------------------------
#def build_file_list(images_root, labels_root, sample_name): 
#    image_sample_root = images_root + '/' + sample_name
#    image_root_len    = len(image_sample_root)
#    label_sample_root = labels_root + '/' + sample_name
#    image_files       = glob(image_sample_root + '/**/*png')
#    file_list         = []
#    for f in image_files:
#        f_relative      = f[image_root_len:]
#        f_dir           = os.path.dirname(f_relative)
#        f_base          = os.path.basename(f_relative)
#        f_base_gt = f_base.replace('leftImg8bit', 'gtFine_color')
#        f_label   = label_sample_root + f_dir + '/' + f_base_gt
#        if os.path.exists(f_label):
#            file_list.append((f, f_label))
#    return file_list


def build_file_list(images_root, labels_root, sample_name): 
    image_sample_root = images_root + '/' + sample_name
    image_root_len    = len(image_sample_root)
    label_sample_root = labels_root + '/' + sample_name
    image_files       = glob(image_sample_root + '/*png')
    label_files       = glob(label_sample_root + '/*png')
    file_list         = []
    for f in range(len(image_files)):
        file_list.append((image_files[f], label_files[f]))

#    if sample_name == 'train':
#       g =  open('D:/file_list_train.txt', 'w')
#       print(file_list, file = g)
#       g.close()

#    if sample_name == 'val':
#       g =  open('D:/file_list_val.txt', 'w')
#       print(file_list, file = g)
#       g.close()
           
    return file_list


#-------------------------------------------------------------------------------
class CityscapesSource:
    #---------------------------------------------------------------------------
    def __init__(self):
#        self.image_size      = (1024, 480)
        self.image_size      = (1120, 1024) #(416, 384)  ##(800,736) 
        self.num_classes     = len(label_defs)

        self.label_colors    = {i: np.array(l.color) for i, l \
                                                     in enumerate(label_defs)}

        self.num_training    = None
        self.num_validation  = None
        self.train_generator = None
        self.valid_generator = None

    #---------------------------------------------------------------------------
    def load_data(self, data_dir, valid_fraction):
        """
        Load the data and make the generators
        :param data_dir:       the directory where the dataset's file are stored
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """
        images_root = data_dir + '/images'
        labels_root = data_dir + '/labels'

        train_images = build_file_list(images_root, labels_root, 'train')
        valid_images = build_file_list(images_root, labels_root, 'val')

        if len(train_images) == 0:
            raise RuntimeError('No training images found in ' + data_dir)
        if len(valid_images) == 0:
            raise RuntimeError('No validatoin images found in ' + data_dir)

        self.num_training    = len(train_images)
        self.num_validation  = len(valid_images)
        self.train_generator = self.batch_generator(train_images)
        self.valid_generator = self.batch_generator(valid_images)

    #---------------------------------------------------------------------------
    def batch_generator(self, image_paths):
        def gen_batch(batch_size, names=False):

#!!!!!!Uwaga wartosci srednie kanalow ze zdjec torow to wpisania
            mean_track_R_channel = 0
            mean_track_G_channel = 0  
            mean_track_B_channel = 0
#------------------------------------------------------------
            meanImagNet_RChannel = 0
            meanImagNet_GChannel = 0
            meanImagNet_BChannel = 0

            
            random.shuffle(image_paths)
            for offset in range(0, len(image_paths), batch_size):
                files = image_paths[offset:offset+batch_size]

                images = []
                labels = []
                names_images = []
                names_labels = []
                for f in files:
                    image_file = f[0]
                    label_file = f[1]                    
                    image = cv2.resize(cv2.imread(image_file), self.image_size)
                    label = cv2.resize(cv2.imread(label_file), self.image_size)
                    
#-------------------------------------------------------------------
                    image = image.astype(np.float32)
                                        
                    image[:, :, 0] = image[:, :, 0]  -  mean_track_B_channel +  meanImagNet_BChannel
                    image[:, :, 1] = image[:, :, 1]  -  mean_track_G_channel +  meanImagNet_GChannel 
                    image[:, :, 2] = image[:, :, 2]  -  mean_track_R_channel +  meanImagNet_RChannel
#--------------------------------------------------------------------                    
                    label_bg   = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
                    label_list = []
                    for ldef in label_defs[1:]:
                        label_current  = np.all(label == ldef.color, axis=2)
                        label_bg      |= label_current
                        label_list.append(label_current)

                    label_bg   = ~label_bg
                    label_all  = np.dstack([label_bg, *label_list])
                    label_all  = label_all.astype(np.float32)

                    images.append(image.astype(np.float32))
                    labels.append(label_all)

                    if names:
                        names_images.append(image_file)
                        names_labels.append(label_file)

                if names:
                    yield np.array(images), np.array(labels), \
                          names_images, names_labels
                else:
                    yield np.array(images), np.array(labels)
        return gen_batch

#-------------------------------------------------------------------------------
def get_source():
  return CityscapesSource()
