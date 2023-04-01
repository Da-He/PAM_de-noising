## PAM de-noising tool

Da He, Jiasheng Zhou, Xiaoyu Shang, Xingye Tang, Jiajia Luo, Sung-Liang Chen

De-Noising of Photoacoustic Microscopy Images by Attentive Generative Adversarial Network ([Paper link](https://ieeexplore.ieee.org/abstract/document/9970755))
- - -
### Citation:
If you find this implementation, the inference tool, or the article is helpful / useful / inspiring, please cite the following :D

@article{he2022noising,\
  title={De-Noising of Photoacoustic Microscopy Images by Attentive Generative Adversarial Network},\
  author={He, Da and Zhou, Jiasheng and Shang, Xiaoyu and Tang, Xingye and Luo, Jiajia and Chen, Sung-Liang},\
  journal={IEEE Transactions on Medical Imaging},\
  year={2022},\
  publisher={IEEE}\
}

---
This implementation was developed based on the repository https://github.com/eriklindernoren/Keras-GAN/tree/master/srgan.


### Dependencies
* tensorflow
* keras
* numpy
* keras_contrib
* scipy==1.2
* Only GPU-based calculation is supported now.

### Inference Utilization
* Step 1: Place all the noisy image (in grayscale .png format only) in a folder (`$input_dir$`)

* Step 2: Under the root folder of the tool, input the command:   
```
python inference.py --input_dir $input_dir$
```

* Step 3: The de-noised results will be place in `$input_dir$/denoised_out`

* Note: Image shapes in the input folder can be arbitrary and different.

### Training Utilization
* Step 1: Prepare the training set and validation set in `.npy` format with 0~1.0 value range. For each set, noisy input data and clean groundtruth data should be placed in `noisy` and `clean` sub-folders individually with the same file names.

* Step 2: Modify Lines `#16`, `#23`, `#55`, `#62` in the "`data_loader.py`" file to specify the dataset.

* Step 3: Hyper-parameters of the training could be modified in Lines `#132`, `#142`, `#147`, `#514` in the "`pam_denoise_main.py`" file.

* Step 4: Run the command ```python pam_denoise_main.py``` to start training.

* Step 5: Training results will be saved in "`./saved_model`".