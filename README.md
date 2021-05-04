# social_distancing_violation_detector
Identify violations of social distancing guidelines in real-time

<!-- Uses Scaled YOLOv4-CSP model from https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp -->
Uses YOLOv4 from https://github.com/Tianxiaomo/pytorch-YOLOv4 (as submodule). 
Uses pruning from https://github.com/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone (as submodule)

### Initializing submodule
To initialize the submodule, clone the repo with  

```
git clone --recurse-submodules https://github.com/snbcypher/social_distancing_violation_detector
```
or 
```
git clone https://github.com/snbcypher/social_distancing_violation_detector
git submodule update --init --recursive
```
