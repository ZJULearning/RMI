#!/bin/bash
# python PATH
export PYTHONPATH="${PYTHONPATH}:${HOME}/github"

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"
# replace comma(,) with empty
#gpus=${gpus//,/}	
# the number of characters
#num_gpus=${#gpus}
#echo "the number of gpus is ${num_gpus}"

# choose the base model
echo ""
echo "0  --  deeplabv3"
echo "1  --  deeplabv3+"
echo "2  --  pspnet"
echo -n "choose the base network: "
read model_choose
case ${model_choose} in
	0 )
		base_model="deeplabv3"
		;;
	1 ) 
		base_model="deeplabv3+"
		;;
	2 )
		base_model="pspnet"
		;;
	* )
		echo "The choice of the segmentation model is illegal!"
		exit 1 
		;;
esac

# choose the backbone
echo ""
echo "0  --  resnet_v1_50"
echo "1  --  resnet_v1_101"
echo "2  --  resnet_v1_152"
echo -n "choose the base network: "
read base_network

case ${base_network} in
	0 )
		backbone="resnet50";;
	1 ) 
		backbone="resnet101";;
	2 )
		backbone="resnet152";;	
	* )
		echo "The choice of the base network is illegal!"
		exit 1 
		;;
esac
echo "The backbone is ${backbone}"
echo "The base model is ${base_model}"


# choose the loss
echo ""
echo "0  --  softmax cross entropy loss."
echo "1  --  sigmoid binary cross entropy loss."
echo "2  --  bce and RMI loss."
echo "3  --  Affinity field loss."
echo "5  --  Pyramid loss."
echo -n "input the loss type of the first stage: "
read loss_type

# choose the dataset
echo ""
echo "0 -- PASCAL VOC2012 dataset"
echo "1 -- Cityscapes"
echo "2 -- CamVid"
echo -n "input the dataset: "
read dataset

# choose the batch size
echo ""
echo -n "input the batch_size (4, 8, 12 or 16): "
read batch_size

if [ ${dataset} = 0 ]
then
	#####################################################
	# SET YOUR DATA DIR HERE 
	#####################################################
	data_dir="${HOME}/dataset/VOCdevkit/VOC2012"
	dataset=pascal
	# !!! train epochs change with batch size ！！！
	crop_size=513
	if [ ${batch_size} = 16 ]
	then
		# first 30K on PASCAL VOC
		train_epochs_1=46
		eval_interval=2
		train_split=trainaug
	elif [ ${batch_size} = 12 ]
	then
		# first 30K on PASCAL VOC
		train_epochs_1=34
		eval_interval=2
		train_split=trainaug
	else
		train_epochs_1=23
		eval_interval=2
		train_split=trainaug	
	fi
elif [ ${dataset} = 1 ]
then
	data_dir="${HOME}/dataset/Cityscapes/"
	dataset=cityscapes
	# 90K on Cityscapes
	if [ ${batch_size} = 8 ]
	then
		crop_size=769
		train_split=train
		train_epochs_1=160
		eval_interval=10
	elif [ ${batch_size} = 4 ]
	then
		crop_size=769
		train_split=train
		train_epochs_1=160
		eval_interval=10
	fi

elif [ ${dataset} = 2 ]
then
	data_dir="${HOME}/dataset/CamVid/"
	dataset=camvid
	# 90K on Cityscapes
	if [ ${batch_size} = 16 ]
	then
		crop_size=481
		train_split=trainval
		train_epochs_1=200
		eval_interval=10
	elif [ ${batch_size} = 4 ]
	then
		crop_size=479
		train_split=trainval
		train_epochs_1=200
		eval_interval=10
	fi
else
	echo "The choice of the dataset is illegal!"
	exit 1 
fi
echo "The data dir is ${data_dir}, the batch size is ${batch_size}."

# learning rate
lr_1=0.007
lr_multiplier=10.0

# slow start
slow_start_steps=1500
slow_start_lr=0.0001

workers=8
accumulation_steps=1

# parameter of rmi
rmi_pool_way=1
rmi_pool_size=4
rmi_pool_stride=4
#rmi_pool_size=2
#rmi_pool_stride=2
rmi_radius=3
loss_weight_lambda=0.5

#####################################################
# STE YOUR MODEL DIR HERE 
#####################################################
pre_dir="rmi_model"
# set the work dir
work_dir="${HOME}/github/RMI"

#####################################################
# STE YOUR RESUME CHECKPOINT HERE 
#####################################################
resume=None

# create PID
case ${loss_type} in
	0 )
		SPID="${pre_dir}/CE_${dataset}_pb${crop_size}-${batch_size}_net${model_choose}-${base_network}_n${num}"
		;;
	1 )
		SPID="${pre_dir}/bce_${dataset}_pb${crop_size}-${batch_size}"
		SPID="${SPID}_net${model_choose}-${base_network}_n${num}"
		;;	
	2 )
		SPID="${pre_dir}/rmi_re_${dataset}_r${rmi_radius}_pw${rmi_pool_way}_st${rmi_pool_stride}_si${rmi_pool_size}"
		SPID="${SPID}_bp${crop_size}-${batch_size}"
		SPID="${SPID}_net${model_choose}-${base_network}-${loss_weight_lambda}_n${num}"		
		;;
	3 )
		SPID="${pre_dir}/affinity_${dataset}_bp${crop_size}-${batch_size}_net${model_choose}-${base_network}_n${num}"		
		;;
	5 ) 
		SPID="${pre_dir}/pyramid_${dataset}_pb${crop_size}-${batch_size}_net${model_choose}-${base_network}_n${num}"
		;;
esac


model_dir=${HOME}/${SPID}
proc_name=${SPID}


# detect the directory
if [ -d ${model_dir} ]
then
	echo "save model into ${model_dir}"
else
	mkdir ${model_dir}
	echo "make the directory ${model_dir}"
fi


# train the model
python ${work_dir}/train.py --backbone ${backbone} \
							--seg_model ${base_model} \
							--slow_start_steps ${slow_start_steps} \
							--slow_start_lr ${slow_start_lr} \
							--init_lr ${lr_1} \
							--lr_multiplier ${lr_multiplier} \
							--model_dir ${model_dir} \
							--workers ${workers} \
							--epochs ${train_epochs_1} \
							--batch_size  ${batch_size} \
							--crop_size ${crop_size} \
							--gpu_ids ${gpus} \
							--checkname deeplab-resnet \
							--dataset ${dataset} \
							--data_dir ${data_dir} \
							--train_split ${train_split} \
							--proc_name ${proc_name} \
							--accumulation_steps ${accumulation_steps} \
							--eval_interval ${eval_interval} \
							--loss_type ${loss_type} \
							--rmi_pool_way ${rmi_pool_way} \
							--rmi_pool_size ${rmi_pool_size} \
							--rmi_radius ${rmi_radius} \
							--rmi_pool_stride ${rmi_pool_stride} \
							--resume ${resume} \
							--loss_weight_lambda ${loss_weight_lambda}

echo "Training Finished!!!"
