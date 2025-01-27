# ZPO Project - Deepness Segmentation Model
![banner](https://github.com/user-attachments/assets/fd38583b-0cd6-4013-8cd7-eebd29a1bb40)

A segmentation model for finding pedestrian and bicycle crossings on satellite images. Based on the examples from [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) by Pavel Iakubovskii.

## Dataset
Model was trained on a dataset of ~1100 satellite images of Poznan, taken from the Poznan 2022 dataset available in https://qms.nextgis.com/. The full dataset is available [here](https://drive.google.com/file/d/1synechPaO5nK8iK8XE_6utxdJ9_LcOLf/view?usp=sharing). In total, there are around 1200 bicycle crossings and 1900 pedestrian crossings annotated. There was no preprocessing performed on the images in dataset. The images are available in **train**, **test** and **valid** folders, and corresponding mask files are located in **trainannot**, **testannot** and **validannot** folders. All images and masks are in png format, and have 512x512 resolution.

## Training
Created model is a segmentation model based on `"FPN"` architecture from the `segmentation_models_pytorch` library. It uses the `"resnext50_32x4k"` encoder. It takes 3 input channels, one for each color component of an RGB image. The output classes are defined as follows:
- **0** - background
- **1** - bicycle-crossing
- **2** - pedestrian-crossing

There are no augmentation methods used. A Dice Loss (from the smp.losses module) is applied in multi-class mode (MULTICLASS_MODE), using logits (i.e., no sigmoid/softmax applied before computing the loss). Adam optimizer is used with a learning rate of 2e-4. For the learning rate scheduler, Cosine Annealing is used. The model is trained for 50 epochs with a batch size of 8, which can be changed in the code for experimentation purposes.

The values of input channels are normalized to range [0, 1] before training.

### Training the model
The code is available in an interactive notebook form. Running the cells in order enables the user to go through the process of loading data, training the model, testing and exporting. The Python version used during development is `3.11`.

## Results
### Example images from the dataset
![dataset](https://github.com/user-attachments/assets/fd52173d-8766-4972-8ce4-20d5782c76df "title")

### Examples of good predictions
![good_predictions](https://github.com/user-attachments/assets/62f8da36-9d69-42de-9ebf-104795e4abe9)

### Examples of bad predictions
![bad_predictions](https://github.com/user-attachments/assets/a5fdf019-9997-4efc-b4d5-13c1ba565fb2)

- IoU for the `train` dataset: 0.995
- IoU for the `test` dataset: 0.987

## Trained model in ONNX ready for `Deepness` plugin
An example trained model in ONNX format can be downloaded [here](https://drive.google.com/file/d/1T9_UnAeZTEYZkS8-OU6t4Sa2MRA9k9z6/view?usp=sharing). It includes all metadata necessary to use in the `Deepness` plugin in `QGIS`.

## Demo instructions and video
![video_example](https://github.com/user-attachments/assets/20868af7-8fdc-4fa1-85f6-32d7bb3ca33d)

The ortophoto used in the demo is `"Poznan 2022 aerial orthophoto high resolution"`, and the location is the crossing near the Politechnika tram stop. Other settings used:
- **resolution** - 10 cm/px
- **batch size** - 1
- **tiles overlap** - 15%
- **class probability threshold** - 0.5
- **dilation/erosion size** - 9 px

## People
- Maciej Brąszkiewicz 147531
- Tomasz Naklicki 147419
