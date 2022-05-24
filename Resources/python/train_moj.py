#!/usr/bin/env python3

#-------------------------------------------------------------------------------

import argparse
import math
import sys
import datetime
import os
import logging, logging.handlers
import logging.config

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import tensorflow as tf
import numpy as np

from fcnvgg import *
from utils import *
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train the FCN')
parser.add_argument('--name', default='C:\\Session\\Learning\\ISR_Xepochs_1120x1024', #lokalizacja nauczonej sieci (wyniku)
                    help='project name')
parser.add_argument('--data-source', default='cityscapes', #dane do uczenia
                    help='data source')
parser.add_argument('--data-dir', default='C:\\Session\\TrainingSet\\', #katalog nadrzędny (images, labels (train, val))
                    help='data directory')
parser.add_argument('--vgg-dir', default= 'C:\\ISR New\\testowanie\\vgg_graph', #lokalizacja sieci początkowej'C:\\SummarySessionMIx_20190722\\Learning\\ISR_46epochs_1120x1024'
                    help='directory for the VGG-16 model')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=2,
                    help='batch size')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--checkpoint-interval', type=int, default=2,
                    help='checkpoint interval')
parser.add_argument('--is_output_printing', type=bool, default=1, 
                    help = 'Is output data printing? (false - only log in file)')

args = parser.parse_args()

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

logs_path = os.getcwd()+"\\logs"

try:
    os.makedirs(logs_path, exist_ok=True)
except OSError:
    print("Creation of the directory "+ logs_path +" failed")

log_time_start = datetime.datetime.now().strftime("%d%m%Y_%H%M")
log_output_name =  logs_path+"\\train_output_" + log_time_start +".log"
setup_logger("log_output", log_output_name)

log_simple_name = logs_path+"\\tain_logs_" + log_time_start +".log"
setup_logger("log_simple", log_simple_name)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def write_to_output(content, always_write_to_output, log_lvl):
    if always_write_to_output or args.is_output_printing:
        print(content)
        logger(content, log_lvl, 'log_output')        
     
    logger(content, log_lvl, 'log_simple')    

#-------------------------------------------------------------------------------
# write_to_output parameters
#-------------------------------------------------------------------------------

write_to_output('[i] Project name:         ' + args.name, 0, 'info')
write_to_output('[i] Data source:          ' + args.data_source, 0, 'info')
write_to_output('[i] Data directory:       ' + args.data_dir, 0, 'info')
write_to_output('[i] VGG directory:        ' + args.vgg_dir, 0, 'info')
write_to_output('[i] # epochs:             ' + str(args.epochs), 0, 'info')
write_to_output('[i] Batch size:           ' + str(args.batch_size), 0, 'info')
write_to_output('[i] Tensorboard directory:' + args.tensorboard_dir, 0, 'info')
write_to_output('[i] Checkpoint interval: ' + str(args.checkpoint_interval), 0, 'info')

try:
    write_to_output('[i] Creating directory {}...'.format(args.name), 0, 'info')
    os.makedirs(args.name)
except (IOError) as e:
    write_to_output('[!]' + str(e), 0, 'error')
    sys.exit(1)

#-------------------------------------------------------------------------------
# Configure the data source
#-------------------------------------------------------------------------------
write_to_output('[i] Configuring data source...', 0, 'info')
try:
    source = load_data_source(args.data_source)
    source.load_data(args.data_dir, 0.1)
    write_to_output('[i] # training samples:   ' + str(source.num_training), 0, 'info')
    write_to_output('[i] # validation samples: ' + str(source.num_validation), 0, 'info')
    write_to_output('[i] # classes:            ' + str(source.num_classes), 0, 'info')
    write_to_output('[i] Image size:           ' + str(source.image_size), 0, 'info')

    train_generator = source.train_generator
    valid_generator = source.valid_generator
    label_colors    = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    write_to_output('[!] Unable to load data source:' + str(e), 1, 'error')
    sys.exit(1)

#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
os.chdir(args.name)
device_name = "/gpu:0"
with tf.device(device_name):
 #with tf.Session() as sess:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        write_to_output('[i] Creating the model...', 0, 'info')
        net = FCNVGG(sess)
        net.build_from_vgg(args.vgg_dir, source.num_classes, progress_hook='tqdm')

        labels = tf.placeholder(tf.float32,
                                shape=[None, None, None, source.num_classes])

        optimizer, loss = net.get_optimizer(labels)

        summary_writer  = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
        saver           = tf.train.Saver(max_to_keep=10)

        label_mapper    = tf.argmax(labels, axis=3)
        n_train_batches = int(math.ceil(source.num_training/args.batch_size))

        initialize_uninitialized_variables(sess)
        write_to_output('[i] Training...', 0, 'info')
        #---------------------------------------------------------------------------
        # Summaries
        #---------------------------------------------------------------------------
        validation_loss = tf.placeholder(tf.float32)
        validation_loss_summary_op = tf.summary.scalar('validation_loss',
                                                    validation_loss)

        training_loss = tf.placeholder(tf.float32)
        training_loss_summary_op = tf.summary.scalar('training_loss',
                                                    training_loss)

        validation_img    = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        validation_img_gt = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        validation_img_summary_op = tf.summary.image('validation_img',
                                                    validation_img)
        validation_img_gt_summary_op = tf.summary.image('validation_img_gt',
                                                        validation_img_gt)
        validation_img_summary_ops = [validation_img_summary_op,
                                    validation_img_gt_summary_op]

        for e in range(args.epochs):
            #-----------------------------------------------------------------------
            # Train
            #-----------------------------------------------------------------------
            generator = train_generator(args.batch_size)
            description = '\n[i] Epoch {:>2}/{}'.format(e+1, args.epochs)        
            training_loss_total = 0
            for x, y in tqdm(generator, total=n_train_batches,
                            desc=description, unit='batches'):
                feed = {net.image_input:  x,
                        labels:           y,
                        net.keep_prob:    0.5}
                loss_batch, _ = sess.run([loss, optimizer], feed_dict=feed)
                training_loss_total += loss_batch * x.shape[0]
            training_loss_total /= source.num_training

            #-----------------------------------------------------------------------
            # Validate
            #-----------------------------------------------------------------------
            generator = valid_generator(args.batch_size)
            validation_loss_total = 0
            imgs          = None
            img_labels    = None
            img_labels_gt = None
            for x, y in generator:
                feed = {net.image_input:  x,
                        labels:           y,
                        net.keep_prob:    1}
                loss_batch, img_classes, y_mapped = sess.run([loss,
                                                            net.classes,
                                                            label_mapper],
                                                            feed_dict=feed)
                validation_loss_total += loss_batch * x.shape[0]

                if imgs is None:
                    imgs          = x[:3, :, :, :]
                    img_labels    = img_classes[:3, :, :]
                    img_labels_gt = y_mapped[:3, :, :]

            validation_loss_total /= source.num_validation

            #-----------------------------------------------------------------------
            # Write loss summary
            #-----------------------------------------------------------------------
            feed = {validation_loss: validation_loss_total,
                    training_loss:   training_loss_total}
            loss_summary = sess.run([validation_loss_summary_op,
                                    training_loss_summary_op],
                                    feed_dict=feed)

            summary_writer.add_summary(loss_summary[0], e)
            summary_writer.add_summary(loss_summary[1], e)

            #-----------------------------------------------------------------------
            # Write image summary every 5 epochs
            #-----------------------------------------------------------------------
            if e % 5 == 0:
                imgs_inferred = draw_labels_batch(imgs, img_labels, label_colors)
                imgs_gt       = draw_labels_batch(imgs, img_labels_gt, label_colors)

                feed = {validation_img:    imgs_inferred,
                        validation_img_gt: imgs_gt}
                validation_img_summaries = sess.run(validation_img_summary_ops,
                                                    feed_dict=feed)
                summary_writer.add_summary(validation_img_summaries[0], e)
                summary_writer.add_summary(validation_img_summaries[1], e)

            #-----------------------------------------------------------------------
            # Save a checktpoint
            #-----------------------------------------------------------------------
            if (e+1) % args.checkpoint_interval == 0:
                checkpoint = '.\\e{}.ckpt'.format(e+1)
                saver.save(sess, checkpoint)
                write_to_output('Checkpoint saved:' + checkpoint, 0, 'info')
                
        checkpoint = '.\\final.ckpt'
        saver.save(sess, checkpoint)
write_to_output('Checkpoint saved:' + checkpoint, 0, 'info')