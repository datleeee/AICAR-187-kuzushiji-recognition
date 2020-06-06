# Kuzushiji Recognition

##### Problem:
* Transcribe this Kuzushiji into contemporary japanese characters, so the Center for Open Data in the Humanities (CODH) can develop better algorithms for Kuzushiji recognition
* Locate and classify each kuzushiji character on a page
 
##### Data: 
* **train/test images** (4 GB)
* **train.csv** - the training set labels and bounding boxes
    * *image_id:* the id code for the image
    * *labels:* a string of all labels for the image. The string should be read as space separated series of values
* **sample_submission.csv** - sample submission file in the correct format
* **unicode_translation.csv** - supplemental file mapping between unicode IDs and Japanese characters

##### Evaluation:
Modified version of the [F1 Score](https://en.wikipedia.org/wiki/F1_score)

##### Submission File:
* For each image in the test set, we must locate and identify all of the kuzushiji characters
* The file should contain a header and have the following format:
    * image_id,labels
    * image_id, {label X Y} {...}

### 1st place solution

**Architecture:** [mmdetection](https://github.com/open-mmlab/mmdetection) in Cascade R-CNN with
* Strong backbones
* Multi-scale train & test

**Data:**
* **Train:** only 1024 x 1024 crops images, because of limited GPU memory
* **Test:** full images (with max size limit)

**Model:** High-resolution network ([HRNet](https://github.com/HRNet/HRNet-Image-Classification))
* LB score 0.935 with:
    * HRNet w32
    * train scales 512-768
    * test scales [0.5, 0.625, 0.75]
* LB score 0.946 with:
    * HRNet w32
    * train scales 768-1280
    * test scales [0.75, 0.875, 1.0, 1.25]
* Ensembling HRNet_w32 and HRNet_w48 => result = 0.950

### 3rd place solution

**Approach:**
2-stage approach + FalsePositive Predictor.

* Detection with Faster R-CNN ResNet101 backbone
* Classification with 5 models with L2-constrained Softmax loss (EfficientNet, SE-ResNeXt101, ResNet152)
* Postprocessing (LightGBM FalsePositive Predictor)

**Detection:** Faster R-CNN (detection)
* ResNet101 backbone
* Multi-scale train&test
* Data augmentation (brightness, contrast, saturation, hue, random grayscale)
* No vertical and horizontal flip 

**Classification:**

1.**Validation strategy**
* Dataset is split train and validation by book titles.

* Generate 2 patterns of training datasets.

    * validation: book_title=200015779 train: others
    * validation: book_title=200003076 train: others
* Characters with few occurrences oversampling at this time.

* For details, see gencsvdenoisedpadtrain_val.py

2.**Preprocessing and data augmentation**
* Preprocessing
    * Denoising and Ben's preprocessing for train and test images, then crop characters
    * When cropping each character, enlarge the area by 5% vertically and horizontally
    * Resize each character to square ignoring aspect ratio
    * To reduce computational resource, some models training with grayscale image
    * To reduce computational resource, undersampling characters that appeared more than 2000 times have been undersampled to 2000 times
* Data augmentation
    * brightness, contrast, saturation, hue, random grayscale, rotate, random resize crop
    * mixup + RandomErasing or ICAP + RandomErasing (details of ICAP will be described later)
    * no vertical and horizontal flip
* Others
    * Use L2-constrained Softmax loss for all models
    * I also tried AdaCos, ArcFace, CosFace, but L2-constrained Softmax was better.
    Warmup learning rate (5epochs)
    * SGD + momentum Optimizer
    * MultiStep LR or Cosine Annealing LR
    * Test-Time-Augmentation 7 crop

**Pseudo labelling**
### Materials
**The dataset:** [Kuzushiji-MNIST](https://www.kaggle.com/anokas/kuzushiji)
**The paper:** [Deep Learning for Classical Japanese Literature](https://arxiv.org/pdf/1812.01718.pdf)

