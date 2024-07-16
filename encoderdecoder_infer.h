#include "onnxruntime_run_options_config_keys.h"
#include "onnxruntime_cxx_api.h"
// #include "t5_logger.h"
#include <spdlog/spdlog.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "sentencepiece_tokenizer.h"

#define max_token 100
#ifdef _WIN32
#define ORTSTRING(str) StrToWstr(str)
#define ORTCHAR(str) StrToWstr(str).c_str()

	inline std::wstring String2wstring(const std::string& str, const std::string& locale)
	{
		typedef std::codecvt_byname<wchar_t, char, std::mbstate_t> F;
		std::wstring_convert<F> strCnv(new F(locale));
		return strCnv.from_bytes(str);
	}

	inline std::wstring  StrToWstr(std::string str) {
		if (str.length() == 0)
			return L"";
		return  String2wstring(str, "zh-CN");

	}

#else

#define ORTSTRING(str) str
#define ORTCHAR(str) str

#endif

class T5EncoderDecoder
    {
    public:
        T5EncoderDecoder(){};
        ~T5EncoderDecoder(){};   
        void Init(const std::string &model_enc,const std::string &model_dec,const std::string & spiece_model, int thread_num);
        std::vector<std::vector<float>> Infer(const std::string &text);
                  
        std::vector<int64_t> PreProcessing(const std::string& text);
        std::string PostProcessing(const std::vector<int>result,const std::vector<int>result_socre);

        std::shared_ptr<Ort::Session> enc_session_ = nullptr;
        std::shared_ptr<Ort::Session> dec_session_ = nullptr;
        Ort::SessionOptions session_options_;
        Ort::Env env_;
        std::vector<const char *> enc_in_names_;
        std::vector<const char *> enc_out_names_;
        std::vector<const char *> dec_in_names_;
        std::vector<const char *> dec_out_names_;
        int sequence_size_;
        void GetInputOutputInfo( const std::shared_ptr<Ort::Session> &session,
        std::vector<const char *> *in_names, std::vector<const char *> *out_names); 
  
    private:
        std::shared_ptr<SentencePieceTokenizer> tokenizer_ = std::make_shared<SentencePieceTokenizer>();
        void ReadModelEnc(const char *model);           
        void ReadModelDec(const char *model); 
        float temperature=0.f;  
        std::vector<Ort::Value> decoder_onnx;  
  
    };