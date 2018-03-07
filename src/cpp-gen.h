#ifndef ANNC_CCP_GEN_H
#define ANNC_CCP_GEN_H

#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "model.h"

namespace annc {

class TensorsHeader {
 public:
  TensorsHeader(Model& model): model_(model) {}

  std::string Assembler();

 private:
  std::string Generate();

  Model& model_;
};

class ModelGen {
 public:
  ModelGen(Model& model): model_(model) {}

  std::string Assembler();

 private:
  std::string Generate();
  std::string GenerateTensorType(const Tensor& tensor, int count);
  std::string GenerateTensorsCode();
  std::string TensorTypeStr(TensorType type);
  std::string ModelGen::TensorCppTypeStr(TensorType type);
  std::string TensorDim(const std::vector<int>& dim);
  float TensorQuantizationScale(const QuantizationParameters& q);
  int TensorQuantizationZeroPoint(const QuantizationParameters& q);
  std::string CheckStatus(const boost::format& msg);

  std::string GenerateOpCode();
  std::string GenerateOpInputs(const std::vector<int>& inputs);
  std::string GenerateOpOutputs(const std::vector<int>& outputs);
  std::string OpTypeStr(BuiltinOperator op_type);

  Model& model_;
};

class CppGen {
 public:
  CppGen(Model& model): model_(model) {}

  void GenFiles(const std::vector<std::string>& namespace_vec,
      const boost::filesystem::path& path);

 private:
  void GenTensorsDataFile(const boost::filesystem::path& path);
  void GenCppFile(const boost::filesystem::path& path);
  Model& model_;
};

}

#endif  // ANNC_CCP_GEN_H