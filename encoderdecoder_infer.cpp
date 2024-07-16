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

int FindMax(float *din, int len)
{
    int i;
    // int len=din.size();
    float max_val = -INFINITY;
    int max_idx = -1;
    for (i = 0; i < len; i++)
    {
        if (din[i] > max_val)
        {
            max_val = din[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// 应用温度采样
void applyTemperatureSampling(std::vector<float> &logits, float temperature)
{
    if (temperature > 0)
    {
        for (float &logit : logits)
        {
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

// 根据logits生成下一个token
int getNextToken(const std::vector<float> &logits, float temperature, int top_k, float top_p, bool greedy_sampling)
{
    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<> dist(logits.begin(), logits.end());
    if (greedy_sampling)
    {
        return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
    }
    else
    {
        return dist(gen);
    }
}

void T5EncoderDecoder::Init(const std::string &model_enc, const std::string &model_dec, const std::string &spiece_model, int thread_num)
{
    session_options_.SetIntraOpNumThreads(thread_num);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options_.DisableCpuMemArena();
    ReadModelEnc(model_enc.c_str());
    ReadModelDec(model_dec.c_str());
    tokenizer_->InitTokenizer(spiece_model);
}

void T5EncoderDecoder::InitV2(const std::string &model, const std::string &spiece_model, int thread_num)
{
    session_options_.SetIntraOpNumThreads(thread_num);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options_.DisableCpuMemArena();
    ReadModelEnc(model.c_str());
    tokenizer_->InitTokenizer(spiece_model);
}

void T5EncoderDecoder::ReadModelEnc(const char *model)
{
    try
    {
        enc_session_ = std::make_shared<Ort::Session>(
            env_, ORTCHAR(model), session_options_);
        LOG_INFO("Successfully load model from{}", model);
    }
    catch (std::exception const &e)
    {

        LOG_ERROR("Error when load T5EncoderDecoder enc onnx model:{}", e.what());
        exit(-1);
    }
    LOG_INFO("GetInputOutputInfo:");
    GetInputOutputInfo(enc_session_, &enc_in_names_, &enc_out_names_);
}

void T5EncoderDecoder::ReadModelDec(const char *model)
{
    try
    {
        dec_session_ = std::make_shared<Ort::Session>(
            env_, ORTCHAR(model), session_options_);
        LOG_INFO("Successfully load model from{}", model);
    }
    catch (std::exception const &e)
    {

        LOG_ERROR("Error when load T5EncoderDecoder dec onnx model:{}", e.what());
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
                  << " dims=" << shape.str() << std::endl;
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
                  << " dims=" << shape.str() << std::endl;
        (*out_names)[i] = name.get();
        name.release();
    }
}
std::vector<int32_t> T5EncoderDecoder::PreProcessing(const std::string &text)
{
    std::vector<int32_t> tokens_32 = tokenizer_->Encode(text);
    tokens_32.push_back(1); // 结束符
    return tokens_32;
}

std::string T5EncoderDecoder::PostProcessing(const std::vector<int> result)
{
    std::string output = tokenizer_->Decode(result);
    ;
    return output;
}

std::vector<std::vector<float>> T5EncoderDecoder::Infer(const std::string &text)
{
    // preprocessing
    std::vector<int32_t> tokens_32 = PreProcessing(text);
    std::vector<int64_t> token_indexs;
    for (const auto &elem : tokens_32)
    {
        token_indexs.push_back(static_cast<int64_t>(elem));
    }

    std::vector<int64_t> shape = {1, (int64_t)token_indexs.size()}; //{1,18}
    auto &input_tensor_values = token_indexs;
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_tensor_values.data(),
                                                                input_tensor_values.size(), shape.data(), 2);

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));

    auto encoder_tensor = enc_session_->Run(Ort::RunOptions{nullptr}, enc_in_names_.data(), ort_inputs.data(),
                                            ort_inputs.size(), enc_out_names_.data(), enc_out_names_.size());

    if (encoder_tensor.size() != enc_out_names_.size())
    {

        LOG_ERROR("error encoder_tensor.size()={}, enc_out_names_.size()={}", encoder_tensor.size(), enc_out_names_.size());
    }
    // get enc_vec
    std::vector<int64_t> enc_shape = encoder_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
    float *enc_data = encoder_tensor[0].GetTensorMutableData<float>();
    std::vector<std::vector<float>> enc_vec(enc_shape[1], std::vector<float>(enc_shape[2]));
    for (int i = 0; i < enc_shape[1]; i++)
    {
        for (int j = 0; j < enc_shape[2]; j++)
        {
            enc_vec[i][j] = enc_data[i * enc_shape[2] + j];
        }
    }
    for (int i = 0; i < 10; i++)
    {
        std::cout << enc_data[i] << ",";
    }
    std::cout << std::endl;

    std::vector<int> new_tokens;
    std::vector<float> new_logits;
    std::vector<int64_t> shape_dec = {1, 1}; //{1,18}
    std::vector<int64_t> values_dec(1, 0);   // 初始化为0的张量值

    std::cout << "dec_in_names_:" << dec_in_names_[0] << std::endl;
    std::cout << "dec_in_names_:" << dec_in_names_[1] << std::endl;
    std::cout << "dec_in_names_.size():" << dec_in_names_.size() << std::endl;
    std::cout << "dec_out_names_.size():" << dec_out_names_.size() << std::endl;
    std::cout << "dec_out_names_:" << dec_out_names_[0] << std::endl;
    std::vector<Ort::Value> decoder_onnx_enc;

    // decoder_onnx_enc.push_back(std::move(encoder_tensor[0]));
    // for(int i=0;i<sequence_size_;i++)
    for (int i = 0; i < 3; i++)
    {
        std::cout << i << std::endl;
        //  for (auto index :shape_dec)
        // {
        //     std::cout<<"shape_dec="<<index<<",";
        // }
        for (int i = 0; i < shape_dec.size(); i++)
        {
            std::cout << "shape_dec i=" << shape_dec[i] << std::endl;
        }
        std::cout << std::endl;
        // for (auto index :values_dec)
        // {
        //     std::cout<<"values_dec="<<index<<",";
        // }
        // std::cout<<std::endl;
        for (int i = 0; i < values_dec.size(); i++)
        {
            std::cout << "values_dec i=" << values_dec[i] << std::endl;
        }
        Ort::Value input_ids_dec = Ort::Value::CreateTensor<int64_t>(memory_info, values_dec.data(),
                                                                     values_dec.size(), shape_dec.data(), 2);
        // decoder_onnx.insert(decoder_onnx.begin(), std::move(input_ids_dec));
        // decoder_onnx.insert(decoder_onnx.begin() + 1, std::move(decoder_onnx_enc[0]));
        // 使用智能指针包装Ort::Value对象
        // std::shared_ptr<Ort::Value> input_tensor_ptr = std::make_shared<Ort::Value>(input_ids_dec);

        // 获取输入张量的数据指针
        float *input_data = input_ids_dec.GetTensorMutableData<float>();

        // 输出输入张量中的值
        for (int i = 0; i < input_ids_dec.GetTensorTypeAndShapeInfo().GetElementCount(); ++i)
        {
            std::cout << "Input value at index " << i << ": " << input_data[i] << std::endl;
        }

        if (decoder_onnx.size() != 2)
        {
            // decoder_onnx.insert(decoder_onnx.begin(), std::move(input_tensor_ptr.get()));
            // decoder_onnx.insert(decoder_onnx.begin(),*input_tensor_ptr);
            decoder_onnx.insert(decoder_onnx.begin(), std::move(input_ids_dec));
            decoder_onnx.insert(decoder_onnx.begin() + 1, std::move(encoder_tensor[0]));
        }
        else
        {
            // decoder_onnx.at(0)=*input_tensor_ptr;
            decoder_onnx.at(0) = std::move(input_ids_dec);
        }
        auto decoder_tensor = dec_session_->Run(Ort::RunOptions{nullptr}, dec_in_names_.data(), decoder_onnx.data(), decoder_onnx.size(), dec_out_names_.data(), dec_out_names_.size());
        try
        {
            // decoder_onnx.erase(decoder_onnx.begin());
            // decoder_onnx.clear();
            int a = 1;
        }
        catch (std::exception const &e)
        {
            std::cerr << "error:" << e.what() << std::endl;
        }
        std::vector<int64_t> dec_shape = decoder_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
        float *dec_data = decoder_tensor[0].GetTensorMutableData<float>();
        // std::cout<<"dec_shape"<<dec_shape[0]<<","<<dec_shape[1]<<","<<dec_shape[2]<<std::endl;
        int vocab_size = dec_shape[2];
        std::vector<float> dec_vec(dec_data, dec_data + vocab_size);
        for (int i = 0; i < 5; i++)
        {
            std::cout << i << "=" << dec_data[i] << std::endl;
        }
        int64_t next_token = (int64_t)FindMax(dec_data, vocab_size);
        if (next_token == 1)
        {
            break;
        }
        // std::cout << "max  value" << "," <<dec_vec[next_token]<< std::endl;
        new_tokens.push_back(next_token);
        values_dec.push_back(next_token);
        shape_dec[1] += 1;
        for (int i = 0; i < new_tokens.size(); i++)
        {
            std::cout << "new_tokens[i]" << "," << new_tokens[i] << std::endl;
        }
        // dec_vec.clear();
        // decoder_tensor.clear();
        // input_tensor_ptr.reset();
    }
    return enc_vec;
}

std::string T5EncoderDecoder::InferV2(const std::string &text)
{
    // preprocessing
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> ort_inputs;

    std::vector<int32_t> token_indexs = tokenizer_->Encode(text);
    token_indexs.push_back(1); // 结束符
    maxseq_length_ = token_indexs.size();

    // 创建输入张量
    const int64_t input_ids_shape[] = {1, (int64_t)token_indexs.size()};
    // std::vector<int32_t> input_ids_data(1 * 18, 1); // 初始化向量
    std::vector<int32_t> max_length_data({maxseq_length_});
    std::vector<int32_t> min_length_data({1});
    std::vector<int32_t> num_beams_data({1});
    std::vector<int32_t> num_return_sequences_data({1});
    std::vector<float> repetition_penalty_data({1.0f});
    std::vector<float> length_penalty_data({1.0f});

    const int64_t max_length_shape[] = {1};
    // std::fill_n(input_ids_data, 1 * 18, 1); // 使用std::fill_n来初始化数组
    Ort::Value input_ids = Ort::Value::CreateTensor<int32_t>(
        memory_info,
        token_indexs.data(), // 数据指针
        token_indexs.size(), // 数据元素数量
        input_ids_shape,     // 形状描述符的指针
        2                    // 形状描述符的维度
    );
    Ort::Value max_length = Ort::Value::CreateTensor<int32_t>(
        memory_info,
        max_length_data.data(), // 注意使用 .data() 方法获取指向vector内部数据的指针
        max_length_data.size(), // vector的大小
        max_length_shape,       // 形状描述符的指针
        1                       // 形状描述符的维度
    );

    Ort::Value min_length = Ort::Value::CreateTensor<int32_t>(
        memory_info,
        min_length_data.data(), // 注意使用 .data() 方法获取指向vector内部数据的指针
        max_length_data.size(), // vector的大小
        max_length_shape,       // 形状描述符的指针
        1);

    Ort::Value num_beams = Ort::Value::CreateTensor<int32_t>(
        memory_info,
        min_length_data.data(), // 注意使用 .data() 方法获取指向vector内部数据的指针
        max_length_data.size(), // vector的大小
        max_length_shape,       // 形状描述符的指针
        1);

    Ort::Value num_return_sequences = Ort::Value::CreateTensor<int32_t>(
        memory_info,
        min_length_data.data(), // 注意使用 .data() 方法获取指向vector内部数据的指针
        max_length_data.size(), // vector的大小
        max_length_shape,       // 形状描述符的指针
        1);

    Ort::Value repetition_penalty = Ort::Value::CreateTensor<float>(
        memory_info,
        repetition_penalty_data.data(), // 注意使用 .data() 方法获取指向vector内部数据的指针
        max_length_data.size(),         // vector的大小
        max_length_shape,               // 形状描述符的指针
        1);

    Ort::Value length_penalty = Ort::Value::CreateTensor<float>(
        memory_info,
        length_penalty_data.data(),
        max_length_data.size(), // vector的大小
        max_length_shape,       // 形状描述符的指针
        1);
    ort_inputs.push_back(std::move(input_ids));
    ort_inputs.push_back(std::move(max_length));
    ort_inputs.push_back(std::move(min_length));
    ort_inputs.push_back(std::move(num_beams));
    ort_inputs.push_back(std::move(num_return_sequences));
    ort_inputs.push_back(std::move(repetition_penalty));
    ort_inputs.push_back(std::move(length_penalty));

    // std::cout << "enc_in_names_:" << enc_in_names_[0] << std::endl;
    // std::cout << "enc_in_names_.size():" << enc_in_names_.size() << std::endl;
    // std::cout << "enc_out_names_.size():" << enc_out_names_.size() << std::endl;
    // std::cout << "enc_out_names_:" << enc_out_names_[0] << std::endl;
    std::vector<Ort::Value> encoder_tensor = enc_session_->Run(Ort::RunOptions{nullptr}, enc_in_names_.data(), ort_inputs.data(),
                                                               ort_inputs.size(), enc_out_names_.data(), enc_out_names_.size());

    if (encoder_tensor.size() != enc_out_names_.size())
    {

        LOG_ERROR("error encoder_tensor.size()={}, enc_out_names_.size()={}", encoder_tensor.size(), enc_out_names_.size());
    }

    // get enc_vec
    std::vector<int64_t> enc_shape = encoder_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int *enc_data = encoder_tensor[0].GetTensorMutableData<int>();

    // print shape
    std::vector<int> decode_result(enc_data, enc_data + maxseq_length_);

    // for (int i = 0; i < maxseq_length_; i++)
    // {
    //     std::cout << enc_data[i] << ",";
    //     // decode_result.push_back(enc_data[i] )
    // }
    // std::cout << std::endl;
    // // decode_result.push_back(1);
    // for (auto index :decode_result)
    // {
    //      std::cout <<index<< ",";
    // }
    // std::cout << std::endl;
    std::string text_result = PostProcessing(decode_result);
    // std::cout <<"text_result:"<<text_result<< std::endl;

    return text_result;
}
