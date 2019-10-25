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
echo -n "choose the base model: "
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
#echo ""
#echo "0  --  resnet_v1_50"
#echo "1  --  resnet_v1_101"
#echo "2  --  resnet_v1_152"
#echo -n "choose the base network: "
#read base_network
base_network=1
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

# choose the batch size
batch_size=1

# choose the dataset
#echo ""
#echo "0 -- PASCAL VOC2012 dataset"
#echo -n "input the dataset: "
#read dataset
dataset=0
if [ ${dataset} = 0 ]
then
	#####################################################
	# SET YOUR DATA DIR HERE 
	#####################################################
	data_dir="${HOME}/dataset/VOCtest"
	dataset=pascal
elif [ ${dataset} = 1 ]
then
	#####################################################
	# SET YOUR DATA DIR HERE 
	#####################################################
	data_dir="${HOME}/dataset/Cityscapes/"
	dataset=cityscapes
else
	echo "The choice of the dataset is illegal!"
	exit 1 
fi
echo "The data dir is ${data_dir}, the batch size is ${batch_size}."


train_split='test'

# set the work dir
work_dir="${HOME}/github/RMI"

# ckpt directory
#####################################################
# STE YOUR CHECKPOINT FILE HERE
#####################################################
#model_name=TBD
#checkpoint_name="deeplab-resnet_ckpt_30406.pth"

#####################################################
# STE YOUR RESUME CKPT HERE
#####################################################
resume=TBD

# model dir and output dir
#####################################################
# STE YOUR MODEL DIR AND OUTPUT DIR HERE 
#####################################################
model_dir=TDB
output_dir=TBD


if [ -d ${output_dir} ]
then
	rm -r  ${output_dir}
	mkdir -p ${output_dir}
	echo "delete and make the directory ${output_dir}"
else
	mkdir -p ${output_dir}
	echo "make the directory ${output_dir}"
fi

# train the model

python ${work_dir}/inference.py --resume ${resume} \
										--seg_model ${base_model} \
										--backbone ${backbone} \
										--model_dir ${model_dir} \
										--train_split ${train_split} \
										--gpu_ids ${gpus} \
										--checkname deeplab-resnet \
										--dataset ${dataset} \
										--data_dir ${data_dir} \
										--output_dir ${output_dir} 
echo "Test Finished!!!"
