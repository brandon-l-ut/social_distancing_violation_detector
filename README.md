# Real Time Social Distancing Detector From Monocular Image
Identify violations of social distancing guidelines in real-time using monocular image.

Uses YOLOv4 from https://github.com/Tianxiaomo/pytorch-YOLOv4 (as submodule). 
Uses pruning from https://github.com/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone (as submodule)

### Initializing submodules from Git
To initialize the submodule, clone the repo with  

```
git clone --recurse-submodules https://github.com/snbcypher/social_distancing_violation_detector
```
or 
```
git clone https://github.com/snbcypher/social_distancing_violation_detector
git submodule update --init --recursive
```
