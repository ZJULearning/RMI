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
echo ""
echo "0  --  resnet_v1_50"
echo "1  --  resnet_v1_101"
echo "2  --  resnet_v1_152"
echo -n "choose the base network: "
read base_network
#base_network=1

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
echo ""
echo "0 -- PASCAL VOC2012 dataset"
echo "1 -- Cityscapes"
echo "2 -- CamVid"
echo -n "input the dataset: "
read dataset

if [ ${dataset} = 0 ]
then
	# data dir
	data_dir="${HOME}/dataset/VOCdevkit/VOC2012"
	checkpoint_name="deeplab-resnet_ckpt_30406.pth"
	train_split='val'
	dataset=pascal
elif [ ${dataset} = 1 ]
then
	data_dir="${HOME}/dataset/Cityscapes/"
	dataset=cityscapes
elif [ ${dataset} = 2 ]
then
	data_dir="${HOME}/dataset/CamVid/"
	checkpoint_name="deeplab-resnet_ckpt_5800.pth"
	dataset=camvid
	train_split='test'
else
	echo "The choice of the dataset is illegal!"
	exit 1 
fi
echo "The data dir is ${data_dir}, the batch size is ${batch_size}."




# set the work dir
work_dir="${HOME}/github/RMI/crf"

# ckpt directory
#####################################################
# STE YOUR CHECKPOINT FILE HERE
#####################################################
model_name=TBD

resume=TBD

# model dir and output dir
model_dir=TBD
output_dir=TBD
if [ -d ${output_dir} ]
then
	echo "save outputs into ${output_dir}"
else
	mkdir -p ${output_dir}
	echo "make the directory ${output_dir}"
fi

# crf steps
# choose the dataset
echo ""
echo -n "input the iteration step of CRF (1 ~ 10):"
read crf_iter_steps
# train the model
#for crf_iter_steps in 5
#do
python ${work_dir}/crf_refine.py --resume ${resume} \
										--seg_model ${base_model} \
										--backbone ${backbone} \
										--model_dir ${model_dir} \
										--train_split ${train_split} \
										--gpu_ids ${gpus} \
										--checkname deeplab-resnet \
										--dataset ${dataset} \
										--data_dir ${data_dir} \
										--crf_iter_steps ${crf_iter_steps} \
										--output_dir ${output_dir} 
#done
echo "Test Finished!!!"
