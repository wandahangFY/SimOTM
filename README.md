# SimOTM: Simplified One-to-Many Preprocessing Method for Object Detection in Grayscale Images

## Introduction
Gray-scale images are widely used in applications such as low-light imaging, medical diagnostics, and industrial inspection due to their simplicity and reduced computational requirements. However, their single-channel nature introduces challenges in object detection, including low object differentiation, noise, and luminance inequality. Traditional preprocessing methods aim to enhance detectability by removing irrelevant information and restoring useful details. Yet, these methods often rely on design for specific scenarios, lack universality, and can even degrade detection results if improperly applied. Consequently, many object detection algorithms avoid extensive preprocessing during training. Additionally, current methods underutilize the potential of single-channel gray-scale images. To address these issues, this paper proposes a simple and general preprocessing algorithm named one to many (OTM) for gray-scale object detection. By converting single-channel gray-scale images into multi-channel formats through image preprocessing and feeding them into the detection model, the algorithm improves detection performance without complex manual design. For validation, a simplified OTM method (SimOTM) is introduced to demonstrate its effectiveness. In this paper, the SimOTM method was incorporated into various object detection frameworks for improving the detection effect of models, and it was tested on four gray object detection datasets from distinct fields. In scenarios where speed remains comparable, the detection performance has been significantly enhanced. Specifically, the mean Average Precision (mAP) of YOLOv5 has improved by 0.43% to 1.37%, YOLOX-s has seen an increase of 0.33% to 3.88%, and YOLOv12 has boosted by 0.69% to 2.29%. 

## Contributions
1)	A novel preprocessing method, OTM-Fusion, is proposed for grayscale object detection.
2)	SimOTM, a simplified version of OTM-Fusion, is introduced for efficient deployment.
3)	The method is integrated and validated across YOLOv3-YOLOv12 models.
4)	Extensive validation on four open-source datasets proves its robustness and generality.


## Quick Start Guide YOLOv11 or YOLOv11-RGBT

### 1. Clone the Project

```bash
git clone https://github.com/ultralytics/ultralytics.git 
cd ultralytics
```

or

```bash
git clone https://github.com/wandahangFY/YOLOv11-RGBT.git 
cd YOLOv11-RGBT
```

### 2. Modify the file
- (1) Replace base.py under YOLOv11 with base.py of this project. (ultralytics/data/base.py)
- (2) Specify the use_simotm parameter of the BaseDataset  (use_simotm="SimOTM")

### 3. Prepare the Dataset
Configure your dataset directory or TXT file .

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Program
```bash
python train.py --data your_dataset_config.yaml
```

### 6. Testing
Run the test script to verify if the data loading is correct:
```bash
python val.py
```

---


## Implemented in  C++ or CUDA
For specific implementation, please refer to Function.cpp. 



## Citation Format
Wan, Dahang & Lu, Rongsheng & Hu, Bingtao & Shen, Siyuan & Xu, Ting & Lang, Xianli. (2023). Otm-Fusion: An Image Preprocessing Method for Object Detection in Grayscale Image. 10.2139/ssrn.4532335. 


## Reference Links
- [Codebase used for overall framework: YOLOv8](https://github.com/ultralytics/ultralytics)
- [Some modules reference from Devil Mask's open-source repository](https://github.com/z1069614715/objectdetection_script)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [Albumentations Data Augmentation Library](https://github.com/albumentations-team/albumentations)


## Closing Remarks
Thank you for your interest and support in this project. The authors strive to provide the best quality and service, but there is still much room for improvement. If you encounter any issues or have any suggestions, please let us know.
Furthermore, this project is currently maintained by the author personally, so there may be some oversights and errors. If you find any issues, feel free to provide feedback and suggestions.

## Other Open-Source Projects
Other open-source projects are being organized and released gradually. Please check the author's homepage for downloads in the future.
[Homepage](https://github.com/wandahangFY)

