# yolov3_detector


![image](https://github.com/Tolashuang/yolov3_detector/blob/master/caea91fd-e457bf7b.jpg)

文档说明：

yolov3_detector是基于darknet的检测图片并输出(类别+目标框位置)坐标代码

编译说明：

1.下载并进入darkent文件夹（https://github.com/pjreddie/darknet.git） & cd ~/darknet/

2.修改darknet中的Makefile并完成编译 & 同时下载yolov3_detector（https://github.com/Tolashuang/yolov3_detector.git）

3.cd到src目录后按照自己需要修改uDetector.cpp文件来满足自己的输出需求

4.make命令编译src中的文件 编译成功后在bin目录下会出现uDetector可执行文件

5.将编译好的darknet中的darknet.h放在src文件夹中 将darknet文件夹中的libdarknet.a拷贝至yolov3_detector的lib文件夹中

使用说明：

1.cd到放有uDetector的文件夹bin目录下

2.输入命令：./uDetect cfgfile weightfile thresh gpuNo imglist outname 

3.PS：outname可以在uDtetector.cpp进行更改 此版是将输出的txt文件和jpg文件放在一起

