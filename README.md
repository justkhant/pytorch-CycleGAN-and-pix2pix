# MANGAClean
MANGAClean uses the PyTorch implementation of the pix2pix GAN network, forked from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to convert raw dirty scans of manga pages into high-quality, denoised images:

The goal of this project was to effectively automate the cleaning process in manga scanlation procedures. But before moving on, here is some background information on what 'scanlation' and what 'manga cleaning' entails: 

## Background
From Wikipedia, **scanlation** is the fan-made scanning, translation, and editing of comics from one language into another. While this is not limited only to manga (Japanese comics), my personal experiences have only been with scanlating manga, so this project has only been trained/tested with manga pages. 

A typical scanlation process has several roles/jobs, but the one we're concerned with is the job of the cleaner. So what exactly does a 'cleaner' do?

First, the cleaner will get the 'raws', which are scans or pictures of the print form of the original content. The quality of the raws differ quite a bit. The job of the cleaner then, is to edit the raws so that the finished products look like officially published online volumes. Below are some examples of raw vs. clean manga pages:

<img src='imgs/Example 1.png' width ="400px"/> <img src='imgs/Example 2.png' width ="400px"/> <img src='imgs/Example 3.png' width ="400px"/> <img src='imgs/Example 4.png' width ="400px"/> 


### The problem of automating the cleaning process

Cleaning a manga page may seem pretty simple, but it is a more complicated and difficult process than just turning a gray scan into black and white. In fact, to be able to clean at a very advanced level and output a very high-quality product takes years of experience and practice. In one of my scanlation groups, cleaners actaully get put through a very rigorous training regime before they can start cleaning real chapters for the group. 

While this may differ betweem different scanlation groups a basic cleaning process looks like this, typically done with Photoshop:

1. **Crop and rotate** - Crop out parts of the raw that are not part of the page. Straighten the page so it is parallel to the image
2. **Adjust Levels** - Adjust levels so the dark parts of the image is black and lighter parts are white
3. **Burn and Dodge** - Darken and lighten parts of the image where needed
4. **Sharpening and Blurring** - Smoothening blemishes and page textures, sharpeing pixels and lines, also done subjectively, 
5. **Topaz Denoise & Clean** - Denoising the image using a Photoshop plugin. Reduces noise and enhances surface texture without losing image detail. 
6. **Repeat Steps 3, 4** - This time more for quality and dust (specks of black dots on white areas and vice vera) removal.
7. **Remove text** - Whitening out text in speech bubbles or outside the panels. 

This is a very rough idea of the process, and again different groups do it with different techniques and tools. And, I'm not a cleaner myself, so I'm probably not doing the process any justice with my crude explanations. To get a better picture of the process check out this youtube video: 
https://www.youtube.com/watch?v=5fyBrsgZb3E&feature=youtu.be \[Credits to Prostyle from MangaStream\]
 
There have probably been many attempts to automate the process. But, I think the most that can be achieved is automating only certain parts of it like level adjusting or denoising. There are definitely some macros that make certain steps easier, like whitening speech bubbles, etc. In fact, Topaz, the photoshop plugin generally used by the scanlation community, is itself an AI powered tool that definitely makes life a lot easier.

However, to automate the WHOLE process from start to finish seems impossible. There are just too many parts that need human input, for instance, recognizing which are folds/lines or blemishes in the paper and which are part of the drawings, or deciding approximating how much to sharpen or blur the images. Moreover, the quality of the raws differ quite a bit, which only increases the subjectivity in the process. 

But perhaps, using deep learning technology, we can _teach_ a network to _learn_ to recognize and learn to do these things. And this is exactly what I'm hoping to experiment with through this project. Will a pix2pix GAN network be able to clean manga up to the same or similar standard to that of humans? 

## Pix2Pix in Pytorch

I decided to try using a pix2pix network because this technology does direct image to image translations. Since we have paired data, that is, we know what the input is and more less or less what the output should look like, it seems pretty fitting for the manga cleaning problem. 

[Below](#implementation) are documentation on the pix2pix implementation in pytorch. It also explains how to setup and train the network using your own data.
This is directly forked from junyanz/pytorch-CycleGAN-and-pix2pix).

## Model 1 Test Results

The first model was trained using about 200 images for training, 20 for validation, and 10 for testing. As for preprocessing, the network only accepts images of dimensions multiple of 4s. So, a 560 x 560 random crop of the training images are applied before feeding them into the network. Below are the results of the training, and what the model currently outputs during testing:  

<img src='imgs/comparisons/labeled/test_result 1.png' width ="800px"/> <img src='imgs/comparisons/labeled/test_result 2.png' width ="800px"/> <img src='imgs/comparisons/labeled/test_result 3.png' width ="800px"/> <img src='imgs/comparisons/labeled/test_result 4.png' width ="800px"/> <img src='imgs/comparisons/labeled/test_result 5.png' width ="800px"/> <img src='imgs/comparisons/labeled/test_result 6.png' width ="800px"/> <img src='imgs/comparisons/labeled/test_result 7.png' width ="800px"/> 

Overall, this current model works great for images without halftones, which are the gray colored sections in the pages. The outputs in these cases are up to the same quality as the actual cleaned version of the raw images, which is great!. However, in the cases where there are halftones, the quality of the output is a lot less that the actual versions. The halftones are not blended well and are very patchy. You can see that in the actual versions, the grays are a lot smoother:

<img src='imgs/comparisons/labeled/halftones 1.png' width ="400px"/>     <img src='imgs/comparisons/labeled/halftones 2.png' width ="400px"/>

To help the network fix these issues, I planned to try collecting and using more data; perhaps double the amount to 400 training images instead of 200 images. I also planned to run it for more epochs! But overall, this model is actaully pretty good enough that a person inexperienced with scalantion and cleaning manga could not tell the difference (i.e. my supervisor). Or, if we didn't care about making the cleaned images very high quality, which some scalantion teams actaully don't really care about, this model would work great, especially since it works very well when the images have no halftones. My end goal is however, to train a model that could output cleaned images that can reach the same quality as the actual versions. 

## Implementation
**Pix2pix:  [Project](https://phillipi.github.io/pix2pix/) |  [Paper](https://arxiv.org/pdf/1611.07004.pdf) |  [Torch](https://github.com/phillipi/pix2pix)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>

**[EdgesCats Demo](https://affinelayer.com/pixsrv/) | [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) | by [Christopher Hesse](https://twitter.com/christophrhesse)**

<img src='imgs/edges2cats.jpg' width="400px"/>

If you use this code for your research, please cite:

Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](http://people.csail.mit.edu/junyanz/projects/pix2pix/pix2pix.bib)

## Talks and Course
pix2pix slides: [keynote](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.key) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.pdf)

## Other implementations

### pix2pix
<p><a href="https://github.com/affinelayer/pix2pix-tensorflow"> [Tensorflow]</a> (by Christopher Hesse),
<a href="https://github.com/Eyyub/tensorflow-pix2pix">[Tensorflow]</a> (by Eyyüb Sariu),
<a href="https://github.com/datitran/face2face-demo"> [Tensorflow (face2face)]</a> (by Dat Tran),
<a href="https://github.com/awjuliani/Pix2Pix-Film"> [Tensorflow (film)]</a> (by Arthur Juliani),
<a href="https://github.com/kaonashi-tyc/zi2zi">[Tensorflow (zi2zi)]</a> (by Yuchen Tian),
<a href="https://github.com/pfnet-research/chainer-pix2pix">[Chainer]</a> (by mattya),
<a href="https://github.com/tjwei/GANotebooks">[tf/torch/keras/lasagne]</a> (by tjwei),
<a href="https://github.com/taey16/pix2pixBEGAN.pytorch">[Pytorch]</a> (by taey16)
</p>
</ul>

### pix2pix
<p><a href="https://github.com/affinelayer/pix2pix-tensorflow"> [Tensorflow]</a> (by Christopher Hesse),
<a href="https://github.com/Eyyub/tensorflow-pix2pix">[Tensorflow]</a> (by Eyyüb Sariu),
<a href="https://github.com/datitran/face2face-demo"> [Tensorflow (face2face)]</a> (by Dat Tran),
<a href="https://github.com/awjuliani/Pix2Pix-Film"> [Tensorflow (film)]</a> (by Arthur Juliani),
<a href="https://github.com/kaonashi-tyc/zi2zi">[Tensorflow (zi2zi)]</a> (by Yuchen Tian),
<a href="https://github.com/pfnet-research/chainer-pix2pix">[Chainer]</a> (by mattya),
<a href="https://github.com/tjwei/GANotebooks">[tf/torch/keras/lasagne]</a> (by tjwei),
<a href="https://github.com/taey16/pix2pixBEGAN.pytorch">[Pytorch]</a> (by taey16)
</p>
</ul>

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org and) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.

### pix2pix train/test
- Download a pix2pix dataset (e.g.[facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)):
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. 
- Train a model:
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
To see more intermediate results, check out  `./checkpoints/facades_pix2pix/web/index.html`.

- Test the model (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- The test results will be saved to a html file here: `./results/facades_pix2pix/test_latest/index.html`. You can find more scripts at `scripts` directory.
- To train and test pix2pix-based colorization models, please add `--model colorization` and `--dataset_mode colorization`. See our training [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization) for more details.

### Apply a pre-trained model (pix2pix)
Download a pre-trained model with `./scripts/download_pix2pix_model.sh`.

- Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_pix2pix_model.sh#L3) for all the available pix2pix models. For example, if you would like to download label2photo model on the Facades dataset,
```bash
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```
- Download the pix2pix facades datasets:
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- Then generate the results using
```bash
python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
```
- Note that we specified `--direction BtoA` as Facades dataset's A to B direction is photos to labels.

- If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use `--model test` option. See `./scripts/test_single.sh` for how to apply a model to Facade label maps (stored in the directory `facades/testB`).

- See a list of currently available models at `./scripts/download_pix2pix_model.sh`

## [Docker](docs/docker.md)
We provide the pre-built Docker image and Dockerfile that can run this code repo. See [docker](docs/docker.md).

## [Datasets](docs/datasets.md)
Download pix2pix/CycleGAN datasets and create your own datasets.

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## [Frequently Asked Questions](docs/qa.md)
Before you post a new question, please first look at the above Q & A and existing GitHub issues.

## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## [Code structure](docs/overview.md)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## Acknowledgments
Our code is forked from [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).
