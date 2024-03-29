# The Pytorch Implementation of Category-consistent deep network learning for accurate vehicle logo recognition 
- This is the offical website for paper ''Category-consistent deep network learning for accurate vehicle logo recognition''. 
- Paper: [Category-consistent deep network learning for accurate vehicle logo recognition](https://www.sciencedirect.com/science/article/abs/pii/S0925231221012145/)
- Authors: [Wanglong Lu](https://longlongaaago.github.io), [Hanli Zhao](http://i3s.wzu.edu.cn/info/1104/1183.htm), [Qi He](http://i3s.wzu.edu.cn/info/1104/1181.htm), [Hui Huang](http://i3s.wzu.edu.cn/info/1104/1163.htm), [Xiaogang Jin](http://www.cad.zju.edu.cn/home/jin/)

## Framework Architecture
![Image](./Images/framework.png#pic_center)

## Requirements
- Pytorch==1.0.1 or higher
- opencv version: 4.1.0

## Datasets
- XMU:
  - Y. Huang, R. Wu, Y. Sun, W. Wang, and X. Ding, Vehicle logo recognition system based on convolutional neural networks with a pretraining strategy, IEEE Transactions on Intelligent Transportation Systems 16 (4) (2015) 1951-1960.
  - https://xmu-smartdsp.github.io/VehicleLogoRecognition.html
- HFUT-VL1 and HFUT-VL2:
  - Y. Yu, J. Wang, J. Lu, Y. Xie, and Z. Nie, Vehicle logo recognition based on overlapping enhanced patterns of oriented edge magnitudes, Computers & Electrical Engineering 71 (2018) 273–283.
  - https://github.com/HFUT-VL/HFUT-VL-dataset
- CompCars:
  - L. Yang, P. Luo, C. C. Loy, and X. Tang, A large-scale car dataset for fine-grained categorization and verification, in: Proc. IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), Boston, MA, 2015, pp.
3973-3981.
  - http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html
- VLD-45:
  - S. Yang, C. Bo, J. Zhang, P. Gao, Y. Li and S. Serikawa, VLD-45: A big dataset for vehicle logo recognition and detection, IEEE Transactions on Intelligent Transportation Systems (2021) doi: 10.1109/TITS.2021.3062113. 
  - https://github.com/YangShuoys/VLD-45-B-DATASET-Detection

## VLF-net for classification (Vehicle logo feature extraction network)

- #### Training with the classification pipeline
  - training XMU dataset
  ```
  python train.py --dataset_name XMU --framework Classification_Network
  ```
  - training HFUT-VL1 dataset
  ```
  python train.py --dataset_name HFUT_VL1 --framework Classification_Network
  ```
  - training HFUT-VL2 dataset
  ```
  python train.py --dataset_name HFUT_VL2 --framework Classification_Network
  ```
  - training CompCars dataset
  ```
  python train.py --dataset_name CompCars --framework Classification_Network
  ```
  - training VLD-45 dataset
  ```
  python train.py --dataset_name VLD-45 --framework Classification_Network
  ```

- #### Testing with the classification pipeline
  - testing XMU dataset
  ```
  python test.py --dataset_name XMU --framework Classification_Network
  ```
  - testing HFUT-VL1 dataset
  ```
  python test.py --dataset_name HFUT_VL1 --framework Classification_Network
  ```
  - testing HFUT-VL2 dataset
  ```
  python test.py --dataset_name HFUT_VL2 --framework Classification_Network
  ```
  - testing CompCars dataset
  ```
  python test.py --dataset_name CompCars --framework Classification_Network
  ```
  - testing VLD-45 dataset
  ```
  python test.py --dataset_name VLD-45 --framework Classification_Network
  ``` 
## VLF-net for category-consistent mask learning
- ### Step 1:
  - Generation of the category-consistent masks. There are more details for the co-localization method [PSOL](https://github.com/tzzcl/PSOL).
  - Please note that we use the generated binary-masks directly instead of the predicted boxes.
  ```
  1. Download the source codes of [PSOL](https://github.com/tzzcl/PSOL).
  2. Add the generate_mask_imagenet.py from our project to the PSOL folder. 
  3. Run the generate_mask_imagenet.py
  - For the generate_mask_imagenet.py
    --input_size    input size for each image (448 for default)
    --gpu           which gpus to use 
    --output_path   the output pseudo boxes folder
    --batch_size    batch_size for executing forward pass of CNN
    --output_mask_path the output pseudo mask folder
  ```
- ### Step 2:
  - After generating the category-consistent masks, we can further organize the training and testing data which are as below:
  ```
  root/
        test/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        train/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
        mask/
            dog/xxx.png
            dog/xxz.png
            cat/123.png
            cat/nsdf3.png
  ```
  Note that each image has the corresponding generated category-consistent mask.
- ### Step 3:
  - Now, you can training the model with the category-consistent mask learning framework 
  - #### Training with the category-consistent deep network learning framework pipeline
      - training XMU dataset
      ```
      python train.py --dataset_name XMU --framework CCML_Network
      ```
      - training HFUT-VL1 dataset
      ```
      python train.py --dataset_name HFUT_VL1 --framework CCML_Network
      ```
      - training HFUT-VL2 dataset
      ```
      python train.py --dataset_name HFUT_VL2 --framework CCML_Network
      ```
      - training CompCars dataset
      ```
      python train.py --dataset_name CompCars --framework CCML_Network
      ```
      - training VLD-45 dataset
      ```
      python train.py --dataset_name VLD-45 --framework CCML_Network
      ```

  - #### Testing with the category-consistent deep network learning framework pipeline
      - testing XMU dataset
      ```
      python test.py --dataset_name XMU --framework CCML_Network
      ```
      - testing HFUT-VL1 dataset
      ```
      python test.py --dataset_name HFUT_VL1 --framework CCML_Network
      ```
      - testing HFUT-VL2 dataset
      ```
      python test.py --dataset_name HFUT_VL2 --framework CCML_Network
      ```
      - testing CompCars dataset
      ```
      python test.py --dataset_name CompCars --framework CCML_Network
      ```
      - testing VLD-45 dataset
      ```
      python test.py --dataset_name VLD-45 --framework CCML_Network
      ``` 

## Experiments
![Image](./Images/Table3.png#pic_center)

![Image](./Images/Table4.png#pic_center)

## Bibtex
- If you find our code useful, please cite our paper:
  ```
  @article{LU2021,
  title = {Category-consistent deep network learning for accurate vehicle logo recognition},
    journal = {Neurocomputing},
    year = {2021},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2021.08.030},
    url = {https://www.sciencedirect.com/science/article/pii/S0925231221012145},
    author = {Wanglong Lu and Hanli Zhao and Qi He and Hui Huang and Xiaogang Jin}
    }
  ```

## Acknowledgements
- The authors would like to thank our anonymous reviewers for their valuable comments to improve this paper.
- Our codes benefit from the official memory-efficient-based [DenseNet](https://github.com/gpleiss/efficient_densenet_pytorch) by [@gpleiss](https://github.com/gpleiss), [Billion-scale-semi-supervised-learning](https://github.com/leaderj1001/Billion-scale-semi-supervised-learning) by [@leaderj1001](https://github.com/leaderj1001), and
 the implementation of the category-consistent from [PSOL](https://github.com/tzzcl/PSOL) by [@tzzcl](https://github.com/tzzcl).
Thanks for their beautiful work.
- Thanks [@kaijieshi7](https://github.com/kaijieshi7), [@Tao Wang](https://taowangzj.github.io/), and [@Changzu Chen](https://github.com/ChenChangzu) for their help in building the project.
