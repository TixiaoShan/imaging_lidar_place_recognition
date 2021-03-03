# Robust Place Recognition using an Imaging Lidar

A place recognition package using high-resolution imaging lidar. For best performance, a lidar equipped with more than 64 uniformly distributed channels is strongly recommended, i.e., [Ouster OS1-128 lidar](https://ouster.com/products/os1-lidar-sensor/).

---

## Dependency

- [ROS](http://wiki.ros.org/ROS/Installation)
- [DBoW3](https://github.com/rmsalinas/DBow3)
  ```
  cd ~/Downloads/
  git clone https://github.com/rmsalinas/DBow3.git
  cd ~/Downloads/DBow3/
  mkdir build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  sudo make install
  ```

---

## Install Package

Use the following commands to download and compile the package.

```
cd ~/catkin_ws/src
git clone https://github.com/TixiaoShan/imaging_lidar_place_recognition.git
cd ..
catkin_make
```

---

## Notes

### Download

The three datasets used in the paper can be downloaded from from [Google Drive](https://drive.google.com/drive/folders/1G1kE8oYGKj7EMdjx7muGucXkt78cfKKU?usp=sharing). The lidar used for data-gathering is Ouster OS1-128.
```
https://drive.google.com/drive/folders/1G1kE8oYGKj7EMdjx7muGucXkt78cfKKU?usp=sharing
```

### Point Cloud Format

The author defined a customized point cloud format, PointOuster, in ```parameters.h```. The customized point cloud is projected onto various images in ```image_handler.h```. If you are using your own dataset, please modify these two files to accommodate data format changes.


### Visualization logic

In the current implementation, the package subscribes to a path message that is published by a SLAM framework, i.e., LIO-SAM. When a new point cloud arrives, the package associates the point cloud with the latest pose in the path. If a match is detected between two point clouds, an edge marker is plotted between these two poses. The reason why it's implemented in this way is that SLAM methods usually suffer from drift. If a loop-closure is performed, the associated pose of a point cloud also needs to be updated. Thus, this visualization logic can update point clouds using the updated path rather than using TF or odometry that cannot be updated later.

### Image Crop

It's recommended to set the ```image_crop``` parameter in ```params.yaml``` to be 196-256 when testing the indoor and handheld datasets. This is because the operator is right behind the lidar during the data-gathering process. Using features extracted from the operator body may cause unreliable matching. This parameter should be set to 0 when testing the Jackal dataset, which improves the reverse visiting detection performance.

---

## Test Package

1. Run the launch file:
```
roslaunch imaging_lidar_place_recognition run.launch
```

2. Play existing bag files:
```
rosbag play indoor_registered.bag -r 3
```

---

## Paper 

Thank you for citing our [paper](./doc/paper.pdf) if you use any of this code or datasets. 
```
@inproceedings{robust2021shan,
  title={Robust Place Recognition using an Imaging Lidar},
  author={Shan, Tixiao and Englot, Brendan and Duarte, Fabio and Ratti, Carlo and Rus Daniela},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={to-be-added},
  year={2021},
  organization={IEEE}
}
```

---

## Acknowledgement

  - The point clouds in the provided datasets are registered using [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM).
  - The package is heavily adapted from [Vins-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono).