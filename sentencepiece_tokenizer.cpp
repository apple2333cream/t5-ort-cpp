#include "sentencepiece_tokenizer.h"
#include <cassert>

void SentencePieceTokenizer::InitTokenizer(const std::string &model_blob)
{

  const auto status = sentence_piece_.Load(model_blob);
    if (!status.ok()) {
       std::cerr << status.ToString() << std::endl;
       // error
    }

}
std::vector<int32_t> SentencePieceTokenizer::Encode(const std::string &text)
{
  std::vector<int32_t> tokens;
  sentence_piece_.Encode(text, &tokens).IgnoreError();
  return tokens;
}
std::vector<std::string> SentencePieceTokenizer::EncodStr(const std::string &text)
{
  std::vector<std::string> tokens;
  sentence_piece_.Encode(text, &tokens).IgnoreError();
  return tokens;
}

std::string SentencePieceTokenizer::Decode(const std::vector<int32_t> &ids)
{
  std::string text;
  sentence_piece_.Decode(ids, &text).IgnoreError();
  return text;
}

size_t SentencePieceTokenizer::GetVocabSize()
{
  auto size = sentence_piece_.GetPieceSize();
  assert(size > 0);
  return size;
}

std::string SentencePieceTokenizer::IdToToken(int32_t id)
{
  return sentence_piece_.IdToPiece(id);
}

int32_t SentencePieceTokenizer::TokenToId(const std::string &token)
{
  return sentence_piece_.PieceToId(token);
}
