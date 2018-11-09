# alexnet-finetune-office-31
Reproduce 60% accuracy on alexnet finetune task on office-31 in Tensorflow.

[Discussion](https://www.qin.ee/2018/06/25/da-office/)

## Data and weights
```python
cd alexnet-finetune-office-31
wget https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
#then download Office-31 from https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code
```
## Run
```python
python3 aw.py
```

## Dependency
Tensorflow 1.x

## Results
Here is the log from one run(Too lazy to use Tensorboard)

```
2018-11-08 10:57:39.388979: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2
2018-11-08 10:57:39.513432: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:895] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-11-08 10:57:39.513878: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1105] Found device 0 with properties:
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.721
pciBusID: 0000:04:00.0
totalMemory: 7.93GiB freeMemory: 7.83GiB
2018-11-08 10:57:39.513900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1070, pci bus id: 0000:04:00.0, compute capability: 6.1)
2018-11-08 10:57:47.592470 Start training...
2018-11-08 10:57:47.592557 Epoch number: 1
train_loss1: 4.2605376 0.001 0.0
train_loss1: 1.5465419 0.0009999987215928159 0.0017045454545454545
2018-11-08 10:58:08.751581 Start validation
2018-11-08 10:58:13.030679 Validation Accuracy = 0.5182
2018-11-08 10:58:13.030747 Saving checkpoint of model...
2018-11-08 10:58:15.724978 Model checkpoint saved at ./checkpoints/model_epoch1.ckpt
2018-11-08 10:58:15.747377 Epoch number: 2
train_loss1: 1.1912441 0.0009999981250041017 0.0025
train_loss1: 0.9063272 0.0009999968466025103 0.004204545454545455
2018-11-08 10:58:35.028907 Start validation
2018-11-08 10:58:38.993618 Validation Accuracy = 0.5484
2018-11-08 10:58:38.993675 Saving checkpoint of model...
2018-11-08 10:58:41.539709 Model checkpoint saved at ./checkpoints/model_epoch2.ckpt
2018-11-08 10:58:41.555288 Epoch number: 3
train_loss1: 0.9984875 0.0009999962500164062 0.005
train_loss1: 1.0593264 0.000999994971620408 0.006704545454545454
2018-11-08 10:59:01.007500 Start validation
2018-11-08 10:59:04.850404 Validation Accuracy = 0.5623
2018-11-08 10:59:04.850461 Saving checkpoint of model...
2018-11-08 10:59:07.583897 Model checkpoint saved at ./checkpoints/model_epoch3.ckpt
2018-11-08 10:59:07.593795 Epoch number: 4
train_loss1: 0.8934074 0.000999994375036914 0.0075
train_loss1: 0.5737518 0.0009999930966465085 0.009204545454545455
2018-11-08 10:59:26.640642 Start validation
2018-11-08 10:59:30.661581 Validation Accuracy = 0.5824
2018-11-08 10:59:30.661635 Saving checkpoint of model...
2018-11-08 10:59:35.997894 Model checkpoint saved at ./checkpoints/model_epoch4.ckpt
2018-11-08 10:59:36.014858 Epoch number: 5
train_loss1: 0.56592584 0.0009999925000656244 0.01
train_loss1: 0.6475414 0.000999991221680812 0.011704545454545455
2018-11-08 10:59:55.269002 Start validation
2018-11-08 10:59:59.176202 Validation Accuracy = 0.5698
2018-11-08 10:59:59.176689 Saving checkpoint of model...
2018-11-08 11:00:01.453169 Model checkpoint saved at ./checkpoints/model_epoch5.ckpt
2018-11-08 11:00:01.472634 Epoch number: 6
train_loss1: 0.58708745 0.000999990625102538 0.0125
train_loss1: 0.41574848 0.0009999893467233184 0.014204545454545456
2018-11-08 11:00:20.563668 Start validation
2018-11-08 11:00:24.428648 Validation Accuracy = 0.5824
2018-11-08 11:00:24.428706 Saving checkpoint of model...
2018-11-08 11:00:27.627665 Model checkpoint saved at ./checkpoints/model_epoch6.ckpt
2018-11-08 11:00:27.637565 Epoch number: 7
train_loss1: 0.53203 0.0009999887501476541 0.015
train_loss1: 0.54651666 0.0009999874717740275 0.016704545454545455
2018-11-08 11:00:46.957242 Start validation
2018-11-08 11:00:50.887759 Validation Accuracy = 0.5899
2018-11-08 11:00:50.888343 Saving checkpoint of model...
2018-11-08 11:00:53.381560 Model checkpoint saved at ./checkpoints/model_epoch7.ckpt
2018-11-08 11:00:53.391878 Epoch number: 8
train_loss1: 0.42048717 0.0009999868752009733 0.0175
train_loss1: 0.68868876 0.0009999855968329393 0.019204545454545457
2018-11-08 11:01:12.605644 Start validation
2018-11-08 11:01:16.524977 Validation Accuracy = 0.5887
2018-11-08 11:01:16.525035 Saving checkpoint of model...
2018-11-08 11:01:21.456430 Model checkpoint saved at ./checkpoints/model_epoch8.ckpt
2018-11-08 11:01:21.466925 Epoch number: 9
train_loss1: 0.3685638 0.0009999850002624952 0.02
train_loss1: 0.52273536 0.000999983721900054 0.021704545454545456
2018-11-08 11:01:40.830874 Start validation
2018-11-08 11:01:44.662195 Validation Accuracy = 0.5811
2018-11-08 11:01:44.662687 Saving checkpoint of model...
2018-11-08 11:01:47.189832 Model checkpoint saved at ./checkpoints/model_epoch9.ckpt
2018-11-08 11:01:47.201358 Epoch number: 10
train_loss1: 0.5281377 0.0009999831253322197 0.0225
train_loss1: 0.46886352 0.0009999818469753712 0.024204545454545454
2018-11-08 11:02:06.447931 Start validation
2018-11-08 11:02:10.355636 Validation Accuracy = 0.5887
2018-11-08 11:02:10.355689 Saving checkpoint of model...
2018-11-08 11:02:12.802450 Model checkpoint saved at ./checkpoints/model_epoch10.ckpt
2018-11-08 11:02:12.813562 Epoch number: 11
train_loss1: 0.37812954 0.0009999812504101469 0.025
train_loss1: 0.4275924 0.000999979972058891 0.026704545454545457
2018-11-08 11:02:31.979294 Start validation
2018-11-08 11:02:35.894304 Validation Accuracy = 0.5874
2018-11-08 11:02:35.894371 Saving checkpoint of model...
2018-11-08 11:02:38.107832 Model checkpoint saved at ./checkpoints/model_epoch11.ckpt
2018-11-08 11:02:38.116953 Epoch number: 12
train_loss1: 0.3709777 0.0009999793754962767 0.0275
train_loss1: 0.4627723 0.0009999780971506134 0.029204545454545455
2018-11-08 11:02:58.536472 Start validation
2018-11-08 11:03:02.499271 Validation Accuracy = 0.5786
2018-11-08 11:03:02.499333 Saving checkpoint of model...
2018-11-08 11:03:04.686087 Model checkpoint saved at ./checkpoints/model_epoch12.ckpt
2018-11-08 11:03:04.686194 Epoch number: 13
train_loss1: 0.30855706 0.0009999775005906089 0.03
train_loss1: 0.30781454 0.0009999762222505381 0.03170454545454545
2018-11-08 11:03:23.755027 Start validation
2018-11-08 11:03:27.729765 Validation Accuracy = 0.5824
2018-11-08 11:03:27.729825 Saving checkpoint of model...
2018-11-08 11:03:29.823449 Model checkpoint saved at ./checkpoints/model_epoch13.ckpt
2018-11-08 11:03:29.832089 Epoch number: 14
train_loss1: 0.3739454 0.0009999756256931433 0.0325
train_loss1: 0.41000128 0.0009999743473586655 0.03420454545454545
2018-11-08 11:03:48.931236 Start validation
2018-11-08 11:03:52.866703 Validation Accuracy = 0.5962
2018-11-08 11:03:52.866756 Saving checkpoint of model...
2018-11-08 11:03:55.048311 Model checkpoint saved at ./checkpoints/model_epoch14.ckpt
2018-11-08 11:03:55.048497 Epoch number: 15
train_loss1: 0.30098462 0.0009999737508038804 0.035
train_loss1: 0.2848215 0.000999972472474995 0.036704545454545455
2018-11-08 11:04:14.023645 Start validation
2018-11-08 11:04:17.957234 Validation Accuracy = 0.6000
2018-11-08 11:04:17.957289 Saving checkpoint of model...
2018-11-08 11:04:22.676109 Model checkpoint saved at ./checkpoints/model_epoch15.ckpt
```
