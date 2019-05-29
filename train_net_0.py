import sys
sys.path.append('/home/iim/mk_utils/mklib/nn/')
import mxnet as mx
from mxconv.mxresnet import MxResNet
from mxiter.mxiter import ImageMultiLabelIter, MyImageIter
from mxloss.mxloss import MxLosses
import numpy as np
from config import net_config as config
import argparse
import logging
import json
import os
from os.path import isdir
from os import mkdir
from shutil import copyfile

# python -m train_net --checkpoints /home/macul/libraries/mk_utils/mx_facerecog_resnet50/output --prefix firstTry
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the checkpoints path
checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
if not isdir(checkpointsPath):
	mkdir(checkpointsPath)

# back net_config file
copyfile('./config/net_config.py', os.path.sep.join([checkpointsPath,'net_config_{}.py'.format(args["start_epoch"])]))
copyfile(__file__, os.path.sep.join([checkpointsPath,os.path.basename(__file__).split('.')[0]+'_{}'.format(args["start_epoch"])+'.py']))

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG, filename=os.path.sep.join([checkpointsPath, "training_{}.log".format(args["start_epoch"])]), filemode="w")

# load the RGB means for the training set, then determine the batch size
means = json.loads(open(config.DATASET_MEAN).read())
batchSize = config.BATCH_SIZE * config.NUM_DEVICES

# create input variable
data = mx.sym.Variable("data")
softmax_label = mx.symbol.Variable('softmax_label')
landmark_gt  = mx.symbol.Variable('landmark_gt')

# construct the training image iterator
'''
trainIter = mx.io.ImageRecordIter(
									path_imgrec=config.TRAIN_MX_REC,
									data_shape=config.Data_Shape,
									batch_size=batchSize,
									rand_mirror=config.Aug_rand_mirror,
									label_width=config.Label_Width,
									mean_r=means["R"],
									mean_g=means["G"],
									mean_b=means["B"],
									scale = 0.0078125,
									shuffle = True,
									preprocess_threads=config.NUM_DEVICES * 2)
trainIter = ImageMultiLabelIter(trainIter, label_len=[1, config.Landmark_Num_Ponits*2], label_name=config.Label_Name)
'''
trainIter = MyImageIter(
						batch_size=batchSize, 
						data_shape=config.Data_Shape,
		                path_imgrec=config.TRAIN_MX_REC,
		                shuffle=True, 
		                mean = [means["R"], means["G"], means["B"]],
		                data_name='data',                  
		                label_name=['softmax_label', 'landmark_gt'],
		                label_len=[1,10],
		                rand_mirror = config.Aug_rand_mirror,
		                cutoff = config.Aug_rand_cutoff,
		                p_glasses = config.Aug_draw_glasses_p,
		                aug_list=config.Aug_list)
trainIter = mx.io.PrefetchingIter(trainIter)
'''
# construct the testing image iterator
testIter = mx.io.ImageRecordIter(
									path_imgrec=config.TEST_MX_REC,
									data_shape=config.Data_Shape,
									batch_size=batchSize,
									label_width=config.Label_Width,
									mean_r=means["R"],
									mean_g=means["G"],
									mean_b=means["B"],
									scale = 0.0078125,
									preprocess_threads=config.NUM_DEVICES * 2)
testIter = ImageMultiLabelIter(testIter, label_len=[1, config.Landmark_Num_Ponits*2], label_name=config.Label_Name)
'''
testIter = MyImageIter(
						batch_size=batchSize, 
						data_shape=config.Data_Shape,
		                path_imgrec=config.TEST_MX_REC,
		                mean = [means["R"], means["G"], means["B"]],
		                data_name='data',                  
		                label_name=['softmax_label', 'landmark_gt'],
		                label_len=[1,10],
		                aug_list=[])
testIter = mx.io.PrefetchingIter(testIter)

# initialize the optimizer
if config.Opt_name == 'SGD':
	opt = mx.optimizer.SGD(	learning_rate=config.Opt_lr, 
							momentum=config.Opt_momentum, 
							wd=config.Opt_weight_decay, 
							rescale_grad=config.Opt_rescale_grad / batchSize)
elif config.Opt_name == 'Adam':
	opt = mx.optimizer.Adam(learning_rate=config.Opt_lr, 
							wd=config.Opt_weight_decay, 
							rescale_grad=config.Opt_rescale_grad / batchSize)


# initialize the model argument and auxiliary parameters
argParams = None
auxParams = None

# if there is no specific model starting epoch supplied, then initialize the network
if args["start_epoch"] <= 0:
	# build the LeNet architecture
	print("[INFO] building network...")
	model, model_landmark = MxResNet.build(	data,
											config.Embedding_Size, 
											config.Resnet_stages, 
											config.Resnet_filters,
											res_ver=config.Resnet_residue_module, 
											in_ver=config.Resnet_input_layer,
											use_se=config.Resnet_use_se)
	with open(os.path.sep.join([checkpointsPath,'{}_deploy_symbol.json'.format(args["prefix"])]),'w') as f:
		f.write(model.tojson())
	model = MxLosses.arc_landmark_loss(	model, 
										model_landmark,
										softmax_label,
										landmark_gt,
										config.Arc_margin_angle, 
										config.Arc_margin_scale, 
										config.NUM_CLASSES, 
										config.Embedding_Size,
										landmark_len=config.Landmark_Num_Ponits*2, 
										grad_scales=[config.Arc_grad_scale, config.Landmark_grad_scale])
# otherwise, a specific checkpoint was supplied
else:
	# load the checkpoint from disk
	print("[INFO] loading epoch {}...".format(args["start_epoch"]))
	_, argParams, auxParams = mx.model.load_checkpoint(os.path.sep.join([checkpointsPath,args["prefix"]]), args["start_epoch"])

	model, model_landmark = MxResNet.build(	data,
											config.Embedding_Size, 
											config.Resnet_stages, 
											config.Resnet_filters,
											res_ver=config.Resnet_residue_module, 
											in_ver=config.Resnet_input_layer,
											bottle_neck=config.Resnet_bottle_neck,
											use_se=config.Resnet_use_se)
	with open(os.path.sep.join([checkpointsPath,'{}_deploy_symbol.json'.format(args["prefix"])]),'w') as f:
		f.write(model.tojson())

	model = MxLosses.arc_landmark_loss(	model, 
										model_landmark,
										softmax_label,
										landmark_gt,
										config.Arc_margin_angle, 
										config.Arc_margin_scale, 
										config.NUM_CLASSES, 
										config.Embedding_Size,
										landmark_len=config.Landmark_Num_Ponits*2,
										grad_scales=[config.Arc_grad_scale, config.Landmark_grad_scale])


# initialize the callbacks and evaluation metrics
batchEndCBs = [mx.callback.Speedometer(batchSize, 250)]
epochEndCBs = [mx.callback.do_checkpoint(os.path.sep.join([checkpointsPath,args["prefix"]]))]
metrics = [	mx.metric.Accuracy(output_names=[model[0].name+'_output'], label_names=[config.Label_Name[0]]), 
			mx.metric.CrossEntropy(output_names=[model[0].name+'_output'], label_names=[config.Label_Name[0]]), 
			mx.metric.MSE(output_names=[model[1].name+'_output'], label_names=[config.Label_Name[1]])]


# construct initializer
if config.Init_name=='Xavier':
	initializer = mx.init.Xavier(rnd_type=config.Init_rnd_type, factor_type=config.Init_factor_type, magnitude=config.Init_magnitude) 
elif config.Init_name=='He':
	initializer = mx.initializer.MSRAPrelu(factor_type=config.Init_factor_type, slope=config.Init_slope)

model = mx.mod.Module(
        context       = [mx.gpu(i) for i in config.DEVICE_IDS],
        symbol        = model,
        label_names   = config.Label_Name
    )

model.fit(	trainIter,
	        begin_epoch        = int(args["start_epoch"]),
	        num_epoch          = config.NUM_EPOCH,
	        eval_data          = testIter,
	        eval_metric        = metrics,
	        kvstore            = 'device',
	        optimizer          = opt,
	        #optimizer_params   = optimizer_params,
	        initializer        = initializer,
	        arg_params         = argParams,
	        aux_params         = auxParams,
	        allow_missing      = True,
	        batch_end_callback = batchEndCBs,
	        epoch_end_callback = epochEndCBs )
			
