#pragma once
#include "sentencepiece_processor.h"
#include <memory>
#include <cassert>
#include "iostream"
class SentencePieceTokenizer
{
public:
  SentencePieceTokenizer(){};
  ~SentencePieceTokenizer(){};
  void InitTokenizer(const std::string &model_blob);
  // int TokenizerUninit();

  std::vector<int32_t> Encode(const std::string &text);
  std::vector<std::string> EncodStr(const std::string &text);
  std::string Decode(const std::vector<int32_t> &ids);
  size_t GetVocabSize();
  std::string IdToToken(int32_t id);
  int32_t TokenToId(const std::string &token);

private:
  // // the tokenizer
  sentencepiece::SentencePieceProcessor sentence_piece_ ;
};
