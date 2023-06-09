# Machine-Learning-Project-2023
This repository is a replication work of ASAPNet (https://github.com/tamarott/ASAPNet) and Pix2PixHD (taken from SPADE https://github.com/NVlabs/SPADE). The purpose is by replicating the two models to make sure that the result stated in the paper is achievable. Both of this models are trained using cityscapes dataset from scratch with image size of 256x512 and the rest of the requirements are kept the same as mentioned in their paper.
After training both models validation was held and inference time, mean Intersection Over Union, and Frechet Inception Distance (FID) is measured.
## ASAPNet generator and Pix2PixHD generator
![image](https://user-images.githubusercontent.com/96078343/227195918-f459b5c3-b4a2-4d9a-86b8-8ce5afe69c15.png)
## Datset preparation
First download the cityscapes dataset from the officail webpage:https://www.cityscapes-dataset.com/downloads/
<pre>
Ground truth zipped: https://www.cityscapes-dataset.com/file-handling/?packageID=3
Input label zipped: https://www.cityscapes-dataset.com/file-handling/?packageID=1
</pre>
There are folders in different city name so merge all data into the folder datasets/cityscapes/: train_images, train_labels, val_images, val_labels accordingly.

## Training the model
For anyone who want to train the model from scratch using the above dataset you can use the following command easily and obtain trained model.
<pre>
python train.py --name [experiment_name] --dataset_mode cityscapes
</pre>
## Testing the model
After training the model or using the pretrained model test can be run using the following simple command on the val dataset. There must be a model inside checkpoints/cityscpes folder to make this inference possible.
<pre>
python test.py --name [name_of_experiment] --dataset_mode cityscapes --batchSize [batch size] --gpu_ids [ids of your gpu]
</pre>

## Metrics
The inference time can be observed during testing of the model after each batch total time taken for prediction will be printed accordingly, in addition to that to obtain the mean Intersection Over Union and Frechet Inception Distance a simple command will give you this output. There is no need to specify the path of a model or validation data, however, you have to make sure that the infered data is saved in the corresponding folder where data is saved during prediction automatically. 
<pre>
python evaluation.py
</pre>
This evlaution method computes meanIoU and FID from saved images, so if anyone wanted to test this method for some other data you can use the argument to specify the path of the data.
<pre>
python evaluation.py --tru_path [rea image path] --pred_path [generated image path] --label_path [mask path]
</pre>

## Additional use of the ASAPNet
As a research direction this model can be used to do tasks such as image construction problems like image denosing and image deblurring, and we have been experiemnting to do such thing in different setup and the result is not attractive but it can be used as a starting point for further study. We have trained  the model in BSD dataset for denosing https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/ and for debluring the dataset from the link https://drive.google.com/file/d/1liKzdjMRHZ-i5MWhC72EL7UZLNPj5_8Y/view?usp=sharing is used. Weights of the model is provided in the link below so you can validate it by just providing the correct path for you testing dataset.\
Denoising weight: https://drive.google.com/file/d/1ucJ3a-W1dSZRXTuOspwm4PnRmXEJl4sC/view?usp=share_link\
Deblurring weight: https://drive.google.com/file/d/1cjuHW2icnk_cIHXJulJGNvBADI-Z-0J5/view?usp=share_link

<pre>
python Debluring.py
python Denoising.py
</pre>
