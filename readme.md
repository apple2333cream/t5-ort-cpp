## t5-ort-cpp
本项目是在CPU环境对谷歌的t5模型用Onnxruntime C++的api进行推理加速，本项目包含了推理demo和http的API服务，支持并发、支持linux/win/arm
若需要GPU环境的加速，请移步至另外一个子项目: https://github.com/apple2333cream/t5-trt-cpp.git  
原始模型仓库：https://huggingface.co/google-t5/t5-base   
### 步骤   
#### 1.环境准备
      onnxruntime 1.15.0
      sentencepiece 0.2.0   
#### 2.模型转换  
    说明：encoder-decoder合并导出处为一个模型  
    git  clone   https://github.com/microsoft/onnxruntime.git  -b  v1.15.0  
    cp -r onnxruntime/python/tools/transformers ./  
    cd  trannsformers  
    python convert_generation.py -m /home/wzp/t5-base --model_type t5 --output /home/wzp/t5-onnx/  
    导出onnx后的python推理示例代码：   python onnx_model_t5.py
   
#### 3.CPP代码编译 
    - 1.mkdir build &&cd  build  
    - 2.cmake ..  
    - 3.make -j8  
#### 4.运行示例   
    demo ./t5_engine --use_mode=0   
    test ./t5_engine --use_mode=1  
    api ./t5_engine --use_mode=2  
    
服务请求示例：
curl -X POST -d "{ \"RequestID\": \"65423221\", \"InputText\": \"translate English to French: I was a victim of a series of accidents.\" }" http://127.0.0.1:17652/T5/register


#### benchmark  
    CPU 内存占用2.2G  
    | thread_num | 时间(ms) |
    |------------|----------|
    | 1          | 1071     |
    | 2          | 706      |
    | 4          | 389      |

#### TODO List 
  - openvino加速

#### Contact Author
qq:807876904 

### 参考
https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/t5