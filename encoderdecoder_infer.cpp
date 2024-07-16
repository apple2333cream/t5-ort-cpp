#include "encoderdecoder_infer.h"
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

template <typename T>
void print_vec_shape(const std::vector<std::vector<T>> &data)
{
    std::cout << "vec_shape= [" << data.size() << ", ";
    if (!data.empty())
    {
        std::cout << data[0].size();
    }
    std::cout << "]" << std::endl;
}

int FindMax(float* din ,int len)
	{
		int i;
        // int len=din.size();
		float max_val = -INFINITY;
		int max_idx = -1;
		for (i = 0; i < len; i++) {
			if (din[i] > max_val) {
				max_val = din[i];
				max_idx = i;
			}
		}
        return max_idx;
	}


// 应用温度采样
void applyTemperatureSampling(std::vector<float>& logits, float temperature) {
    if (temperature > 0) {
        for (float& logit : logits) {
            logit /= temperature;
        }
    }
}

// // 应用重复惩罚
// void applyRepetitionPenalty(std::vector<float>& logits, const std::vector<int64_t>& generated, float repetition_penalty) {
//     std::unordered_set<int64_t> generated_set(generated.begin(), generated.end());
//     for (int64_t token : generated_set) {
//         logits[token] /= repetition_penalty;
//     }
// }


template<typename T>
int argmax(const std::vector<T>& vec) {
    if (vec.empty()) {
        throw std::invalid_argument("Vector is empty");
    }

    // 找到向量中最大元素的迭代器
    auto result = std::max_element(vec.begin(), vec.end());

    // 计算并返回迭代器相对于向量开始位置的偏移量
    return std::distance(vec.begin(), result);
}

// 根据logits生成下一个token
int getNextToken(const std::vector<float>& logits, float temperature, int top_k, float top_p, bool greedy_sampling) {
    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<> dist(logits.begin(), logits.end());
    if (greedy_sampling) {
        return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
    } else {
        return dist(gen);
    }
}

void T5EncoderDecoder::Init(const std::string &model_enc,const std::string &model_dec, const std::string &spiece_model, int thread_num)
{
    session_options_.SetIntraOpNumThreads(thread_num);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options_.DisableCpuMemArena();
    ReadModelEnc(model_enc.c_str());
    ReadModelDec(model_dec.c_str());
    tokenizer_->InitTokenizer(spiece_model);
}

void T5EncoderDecoder::ReadModelEnc(const char *model)
{
    try
    {
        enc_session_ = std::make_shared<Ort::Session>(
            env_, ORTCHAR(model), session_options_);
        SPDLOG_INFO("Successfully load model from{}", model);
    }
    catch (std::exception const &e)
    {

        SPDLOG_ERROR("Error when load T5EncoderDecoder enc onnx model:{}", e.what());
        exit(-1);
    }
    GetInputOutputInfo(enc_session_, &enc_in_names_, &enc_out_names_);
}

void T5EncoderDecoder::ReadModelDec(const char *model)
{
    try
    {
        dec_session_ = std::make_shared<Ort::Session>(
            env_, ORTCHAR(model), session_options_);
        SPDLOG_INFO("Successfully load model from{}", model);
    }
    catch (std::exception const &e)
    {

        SPDLOG_ERROR("Error when load T5EncoderDecoder dec onnx model:{}", e.what());
        exit(-1);
    }
    GetInputOutputInfo(dec_session_, &dec_in_names_, &dec_out_names_);

}

 void T5EncoderDecoder::GetInputOutputInfo(
        const std::shared_ptr<Ort::Session> &session,
        std::vector<const char *> *in_names, std::vector<const char *> *out_names)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        // Input info
        int num_nodes = session->GetInputCount();
        in_names->resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
        {
            std::unique_ptr<char, Ort::detail::AllocatedFree> name = session->GetInputNameAllocated(i, allocator);
            Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            std::vector<int64_t> node_dims = tensor_info.GetShape();
            std::stringstream shape;
            for (auto j : node_dims)
            {
                shape << j;
                shape << " ";
            }
            std::cout << "\tInput " << i << " : name=" << name.get() << " type=" << type
                      << " dims=" << shape.str()<<std::endl;
            (*in_names)[i] = name.get();
            name.release();
        }
        // Output info
        num_nodes = session->GetOutputCount();
        out_names->resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
        {
            std::unique_ptr<char, Ort::detail::AllocatedFree> name = session->GetOutputNameAllocated(i, allocator);
            Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            std::vector<int64_t> node_dims = tensor_info.GetShape();
            std::stringstream shape;
            for (auto j : node_dims)
            {
                shape << j;
                shape << " ";
            }
            std::cout << "\tOutput " << i << " : name=" << name.get() << " type=" << type
                      << " dims=" << shape.str()<<std::endl;
            (*out_names)[i] = name.get();
            name.release();
        }
    }
std::vector<int64_t> T5EncoderDecoder::PreProcessing(const std::string &text)
{
    std::vector<int32_t> tokens_32 = tokenizer_->Encode(text);
    std::vector<int64_t> tokens_64; 
    // 将vec32的内容转换为int64_t并插入到vec64中
    for (const auto& elem : tokens_32) {
        tokens_64.push_back(static_cast<int64_t>(elem));
    }
    int token_size=tokens_64.size();
    tokens_64.push_back(1);//结束符
    std::cout<<"tokens_64:"<<token_size<<std::endl;
    sequence_size_=token_size;
    for (auto token : tokens_64)
    {
        std::cout << token << "|";
    }
    std::cout << std::endl;
    return tokens_64;
}

    std::string T5EncoderDecoder::PostProcessing(const std::vector<int>result,const std::vector<int>result_socre)
     {
        std::string output;
        return output;
     }


 std::vector<std::vector<float>> T5EncoderDecoder::Infer(const std::string &text)
{
    // preprocessing
    std::vector<int64_t> token_indexs = PreProcessing(text);
    std::vector<int64_t> shape = {1, (int64_t)token_indexs.size()}; //{1,18}
    auto &input_tensor_values = token_indexs;
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_tensor_values.data(),
                                                                input_tensor_values.size(), shape.data(), 2);

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    std::cout<<"enc_in_names_:"<<enc_in_names_[0]<<std::endl;
    std::cout<<"enc_in_names_.size():"<<enc_in_names_.size()<<std::endl;
    std::cout<<"enc_out_names_.size():"<<enc_out_names_.size()<<std::endl;
    std::cout<<"enc_out_names_:"<<enc_out_names_[0]<<std::endl;
    auto encoder_tensor = enc_session_->Run(Ort::RunOptions{nullptr}, enc_in_names_.data(), ort_inputs.data(),
                                      ort_inputs.size(), enc_out_names_.data(), enc_out_names_.size());

    if (encoder_tensor.size() != enc_out_names_.size())
    {
        printf(" error encoder_tensor.size()=%d, enc_out_names_.size()=%d\n", encoder_tensor.size(), enc_out_names_.size());
      
    }
    // get enc_vec
    std::vector<int64_t> enc_shape = encoder_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
    float* enc_data = encoder_tensor[0].GetTensorMutableData<float>();
    std::vector<std::vector<float>> enc_vec(enc_shape[1], std::vector<float>(enc_shape[2]));
    for (int i = 0; i < enc_shape[1]; i++) {
        for (int j = 0; j < enc_shape[2]; j++) {
            enc_vec[i][j] = enc_data[i * enc_shape[2] + j];
        }
    }
    for (int i = 0; i < 10; i++)
    {
        std::cout << enc_data[i] << ",";
    }
    std::cout << std::endl;   

    std::vector<int> new_tokens;
    std::vector<float>new_logits;
    std::vector<int64_t> shape_dec = {1,1}; //{1,18}
    std::vector<int64_t> values_dec(1, 0); // 初始化为0的张量值

    std::cout<<"dec_in_names_:"<<dec_in_names_[0]<<std::endl;
    std::cout<<"dec_in_names_:"<<dec_in_names_[1]<<std::endl;
    std::cout<<"dec_in_names_.size():"<<dec_in_names_.size()<<std::endl;
    std::cout<<"dec_out_names_.size():"<<dec_out_names_.size()<<std::endl;
    std::cout<<"dec_out_names_:"<<dec_out_names_[0]<<std::endl;
    std::vector<Ort::Value> decoder_onnx_enc;  

    // decoder_onnx_enc.push_back(std::move(encoder_tensor[0]));
    // for(int i=0;i<sequence_size_;i++)
    for(int i=0;i<3;i++)
    {       
         std::cout<<i<<std::endl;
        //  for (auto index :shape_dec)
        // {
        //     std::cout<<"shape_dec="<<index<<",";
        // }
        for (int i =0;i<shape_dec.size();i++)
        {
            std::cout<<"shape_dec i="<<shape_dec[i]<<std::endl;
        }
          std::cout<<std::endl;
        // for (auto index :values_dec)
        // {
        //     std::cout<<"values_dec="<<index<<",";
        // }
        // std::cout<<std::endl;
        for (int i =0;i<values_dec.size();i++)
        {
            std::cout<<"values_dec i="<<values_dec[i]<<std::endl;
        }
        Ort::Value input_ids_dec = Ort::Value::CreateTensor<int64_t>(memory_info, values_dec.data(),
                                                                values_dec.size(), shape_dec.data(), 2);
        // decoder_onnx.insert(decoder_onnx.begin(), std::move(input_ids_dec));  
        // decoder_onnx.insert(decoder_onnx.begin() + 1, std::move(decoder_onnx_enc[0]));
           // 使用智能指针包装Ort::Value对象
        // std::shared_ptr<Ort::Value> input_tensor_ptr = std::make_shared<Ort::Value>(input_ids_dec);
  

        // 获取输入张量的数据指针
        float* input_data = input_ids_dec.GetTensorMutableData<float>();

        // 输出输入张量中的值
        for (int i = 0; i < input_ids_dec.GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
            std::cout << "Input value at index " << i << ": " << input_data[i] << std::endl;
        }

 
        if (decoder_onnx.size()!=2)
        {
            // decoder_onnx.insert(decoder_onnx.begin(), std::move(input_tensor_ptr.get())); 
            // decoder_onnx.insert(decoder_onnx.begin(),*input_tensor_ptr); 
            decoder_onnx.insert(decoder_onnx.begin(),std::move(input_ids_dec)); 
            decoder_onnx.insert(decoder_onnx.begin() + 1, std::move(encoder_tensor[0]));
        }
        else{
            // decoder_onnx.at(0)=*input_tensor_ptr;
            decoder_onnx.at(0)=std::move(input_ids_dec);
        }
        auto decoder_tensor = dec_session_->Run(Ort::RunOptions{ nullptr }, dec_in_names_.data(), decoder_onnx.data(), decoder_onnx.size(), dec_out_names_.data(), dec_out_names_.size());
        try
        {   
            // decoder_onnx.erase(decoder_onnx.begin());
            // decoder_onnx.clear();
            int a=1;
        }
        catch (std::exception const &e)
        {
            std::cerr<<"error:"<<e.what()<<std::endl;          
        }
        std::vector<int64_t> dec_shape = decoder_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
        float* dec_data = decoder_tensor[0].GetTensorMutableData<float>();
        // std::cout<<"dec_shape"<<dec_shape[0]<<","<<dec_shape[1]<<","<<dec_shape[2]<<std::endl;
        int vocab_size=dec_shape[2];
        std::vector<float>dec_vec(dec_data,dec_data+vocab_size);
        for (int i=0;i<5;i++)
        {
            std::cout<<i<<"="<<dec_data[i]<<std::endl;
        }
        int64_t next_token =(int64_t) FindMax(dec_data,vocab_size);
        if (next_token==1)
        {
            break;
        }       
        // std::cout << "max  value" << "," <<dec_vec[next_token]<< std::endl;
        new_tokens.push_back(next_token);
        values_dec.push_back(next_token);
        shape_dec[1] += 1;
        for (int i = 0; i < new_tokens.size(); i++)
        {
            std::cout << "new_tokens[i]" << "," <<new_tokens[i]<< std::endl;
        }
        // dec_vec.clear();
        // decoder_tensor.clear();
        // input_tensor_ptr.reset();
    }    
    return enc_vec;

    
}
