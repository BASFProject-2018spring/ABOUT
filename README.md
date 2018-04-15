# ABOUT

This organization account is to host all softwares / tools / modules we develop for the BASF nematode project.

# Working Pipeline

Unless mentioned, `python` stands for `python3`.

## Obtain training/validation data

Use [BBox-Label-Tool-large-img](https://github.com/BASFProject-2018spring/BBox-Label-Tool-large-img). For Windows users, you can download the "compiled" `.exe` in release.

This tool will produce a `Labels` folder containing all label txts with same relative paths as original bmp images. Now you should have `Images` and `Labels` folder with corresponding images and labels.

## Data cleaning

There will be multiple subfolders in both `Images` and `Labels`. Use the [preprocess script](https://github.com/BASFProject-2018spring/VOC_format_builder/blob/master/preprocess_imgs_rcnn.py) to get all images under a single directory and rename them into `NUMBER.jpg` (convert to `.jpg`). The command is:

```bash
python preprocess_imgs_rcnn.py --old_img_folder Images --old_label_folder Labels --new_img_folder new_img --new_label_folder new_lbl --quality 95
```

After this step, you will have two folders `new_img` and `new_lbl`.

## Data augmentation (Optional)

Rotation and horizontal flip will give you 4x as many data. Illumination augmentation will give you 3x as many data. A combination gives you 12x as many data.

Use scripts [here](https://github.com/BASFProject-2018spring/Data_augmentation) to do data augmentation. The command is:

```bash
python rotate_flip.py --img_folder new_img --label_folder new_lbl --aug_img_folder aug_img --aug_label_folder aug_lbl --quality 95
python illumination.py --img_folder aug_img --label_folder aug_lbl --aug_img_folder aug1_img --aug_label_folder aug1_lbl --quality 95
```

For consistency on folder names, in this demonstration, I renamed the folder names to `new_*`

```bash
rm -rf new_lbl
rm -rf new_img
rm -rf aug_img
rm -rf aug_lbl
mv aug1_img new_img
mv aug1_lbl new_lbl
```

## Build VOC-like datasets

Use the script [here](https://github.com/BASFProject-2018spring/VOC_format_builder/blob/master/voc_dataset_build.py)

The command is (you should specify N and V):

```bash
python voc_dataset_build.py --voc_folder VOC2007 --img_folder new_img --label_folder new_lbl --test_num N --val_num V
```

If you did data augmentation, the validation set is NOT TRUE (since validation data are seen during training). We set test_num (i.e. N) to 1 since we have seperate test data.

After this step, you should have folder `VOC2007`. Suppose the parent folder **absolute path** is `X` (i.e. you have path `X/VOC2007`)

## Train the network

You need a GPU and **Python 2.7**. 

The data setup is:

```bash
git clone https://github.com/BASFProject-2018spring/faster-RCNN-gpu.git
cd faster-RCNN-gpu/data
ln -sf X/VOC2007 VOCdevkit2007
```

Next follow the instructions in [this repo](https://github.com/BASFProject-2018spring/faster-RCNN-gpu). We only kept training scripts from original Faster-RCNN implementation in this repository.

## Test on GPU

You need **Python 2.7**

You will see `run.py` under `tools` folder in `faster-RCNN-gpu` repo.

Get the model from `output` folder and put it to `$HOME/models`, rename the model to either `res152.pth`, `res101.pth` or `vgg16.pth` according to which model you were training.

put all your test images under `$HOME/imgwork/input` and run command:

```bash
python run.py --net NET
```

(NET can be either vgg16, res101 or res152)

In `$HOME/imgwork/output`, you will see all annotated images and the detailed boxes as txt files (box coordinates, classifications and confidences). If you can code, you may want to process those txt files by your own (very simple). Or you may try our GUI (details later)...

## Test on CPU

You need **Python 2.7**

Use [this repo](https://github.com/BASFProject-2018spring/faster-RCNN-cpu) instead. You'll see `tools/run.py` as well but it uses CPU.

## GUI, auto-updates, CSV export, R^2 calculations...

This is just for the convenience of clients so they don't need to write additional code to process the inference txts. Documents are [here](https://github.com/BASFProject-2018spring/GUI).

# License

Unless mentioned, all modules for this project are under **GNU GPL v3** license.
