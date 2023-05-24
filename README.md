## 功能

```shell
深度图生成点云图
1）生成点云数据 *.ply, *.pcd, *.txt格式
2）对左右图进行立体校正
```

## 编译

```shell
mkdir build
cd build
cmake ..
make -j4
```

## 数据配置
```shell
修改宏定义目录
校准参数文件
#define CALIB_PATH "../wall/calib.yaml"  
左右图和深度文件
#define IMAGE_PATH "../wall/"            
```

## 功能配置
```shell
对Realsense文件进行点云转换
#define REALSENSE_ON  
开启点云滤波器  
#define SPATIAL_FILTER  
```
