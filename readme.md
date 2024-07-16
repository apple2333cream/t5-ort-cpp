t5 onnxruntime的cxx api推理  

步骤  
1.mkdir build &&cd  build  
2.cmake ..  
3.make -j8  
4.运行示例   
demo ./t5_engine --use_mode=0   
test ./t5_engine --use_mode=1  
api ./t5_engine --use_mode=2  
benchmark  
CPU 内存占用2.2G  
| thread_num | 时间(ms) |
|------------|----------|
| 1          | 1071     |
| 2          | 706      |
| 4          | 389      |
| 8          | 276      |

服务请求示例：

curl -X POST -d "{ \"RequestID\": \"65423221\", \"InputText\": \"translate English to French: I was a victim of a series of accidents.\" }" http://127.0.0.1:17652/T5/register