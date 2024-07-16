#include <sentencepiece_processor.h>
#include "sentencepiece_tokenizer.h"
#include "encoderdecoder_infer.h"
#include "iostream"
#include <memory>
#include <spdlog/spdlog.h>

int main()
{    
    // std::string spiece_model = "/home/wzp/t5-onnx/spiece.model";
    // std::shared_ptr<SentencePieceTokenizer> sentence_hanle = std::make_shared<SentencePieceTokenizer>();
    // sentence_hanle->InitTokenizer(spiece_model); 
    // std::vector<int32_t> ids;
    // std::vector<std::string> pieces;
    // std::string text = "translate English to French: I was a victim of a series of accidents.";
    // ids = sentence_hanle->Encode(text);
    // pieces = sentence_hanle->EncodStr(text);
    // for (const int id : ids)
    // {
    //     std::cout << id << "|";
    // }
    // std::cout << std::endl;
    // for (const std::string &token : pieces)
    // {
    //     std::cout << token << "|";
    // }
    // std::cout << std::endl;
    std::string enc_model_path="/home/wzp/t5-onnx/t5-encoder-12.onnx";
    std::string dec_model_path="/home/wzp/t5-onnx/t5-decoder-with-lm-head-12.onnx";
    std::string spiece_model_path="/home/wzp/t5-onnx/spiece.model";
    int thread_num=1;
    std::string text = "translate English to French: I was a victim of a series of accidents.";
    std::shared_ptr<T5EncoderDecoder>encoder_handle = nullptr;
    encoder_handle = std::make_shared<T5EncoderDecoder>();
    encoder_handle->Init(enc_model_path,dec_model_path,spiece_model_path,thread_num);
    //  std::vector<std::vector<float>> out_prob;
     std::vector<std::vector<float>> out_prob;
    out_prob=encoder_handle->Infer(text);
    //  FUNASR_HANDLE sv_hanlde = CamPPlusSvInit(model_path, thread_num);
    SPDLOG_INFO("encoder 模型加载成功！");

    return 0;
}
