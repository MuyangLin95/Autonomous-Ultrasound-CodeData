# Deep learning model and logistic model for ultrasound image classification

The deep learning models used in this porject include classificaiton models (VGG11, VGG13, ResNet, MobileNet) and Minimal-Entropy Correlation Alignment (MECA) model (see the Reference section for further details). 

The logistic model used in this project is based on conventional image processing methods. The algorithm was developed with MATLAB.


## Reference
The classificaiton models (VGG11, VGG13, ResNet, MobileNet) can be refered to pytorch library.
The Minimal-Entropy Correlation Alignment model based off of the original repository at
[minimal-entropy-correlation-alignment](https://github.com/pmorerio/minimal-entropy-correlation-alignment).

**"Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain Adaptation"**  
Pietro Morerio, Jacopo Cavazza and Vittorio Murino  
*International Conference on Learning Representations (ICLR), 2018*  
[PDF](https://openreview.net/forum?id=rJWechg0Z)

      @article{
      morerio2018minimalentropy,
      title={Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain Adaptation},
      author={Morerio, Pietro and Cavazza, Jacopo and Murino, Vittorio},
      journal={International Conference on Learning Representations},
      year={2018},
      url={https://openreview.net/forum?id=rJWechg0Z},
      }

## Code
The classificaiton models (VGG11, VGG13, ResNet, MobileNet) can be directly called from pytorch library.

The 'meca' directory contains the code needed to train and run the model - follow the README.md file in the subdirectory for instructions.

The provided code runs with Python 3.8. For the installation of ``tensorflow-gpu`` please refer to the [website](http://www.tensorflow.org/install/).

The following command should install the main dependencies on most Linux (Ubuntu) machines

```
git checkout python3
sudo apt-get install python3-dev python3-pip
sudo pip3 install -r requirements.txt
```

The logistic model scripts needs to run with MATLAB 2020b. Scripts "logistic_model_for_ultrasound_image_classification.m" and "logistic_model_step_by_step_processing.mlx" needs to run in the same directory with the image folders.

## Dataset access
The image data sets for classification models can be accessed via google drive (https://drive.google.com/drive/folders/1fAZPEFKwvl3VbpdwvY9UjHFVOXOMi4dR?usp=share_link).

The image data sets for MECA model can be accessed via google drive (https://drive.google.com/drive/folders/1DoazVlxnZeUji9mSJ90ihH9jGDrQVtJH?usp=share_link).

The image datasets for logistic model can be accessed via google drive (https://drive.google.com/drive/folders/1vdOLWwQfscgyGoMbCjSRBj5tTOdjUCJ4?usp=share_link).

## License
This repository is released under the MIT LICENSE.
