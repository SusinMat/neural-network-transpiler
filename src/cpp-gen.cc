#include <iostream>
#include <sstream>
#include <fstream>

#include "cpp-gen.h"

namespace annc {

std::string TensorsHeader::Generate() {
  const std::vector<Buffer>& buffers = model_.Buffers();
  std::string str_code = "";

  size_t i = 0;
  for (const auto& buf : buffers) {
    std::stringstream ss;
    ss << "const char tbuf_" << i << "[] = {";

    if (buf.Data().size() > 0) {
      std::string str_vec = "";
      for (const auto& c : buf.Data()) {
        str_vec += std::to_string(static_cast<int>(c)) + ",";
      }

      // remove the last ',' char
      str_vec = str_vec.substr(0, str_vec.length() - 1);
      ss << str_vec;
    }

    ss << "};\n";
    str_code += ss.str();
    ++i;
  }

  return str_code;
}

std::string TensorsHeader::Assembler(
    const std::vector<std::string>& namespace_vec) {
  size_t namespace_count = namespace_vec.size();
  std::string str_code = "";

  for (const auto& name : namespace_vec) {
    str_code += name + "{";
  }

  str_code += "\n";

  str_code += Generate();

  for (size_t i = 0; i < namespace_count; i++) {
    str_code += "}";
  }

  str_code += "\n";

  return str_code;
}

std::string ModelGen::Generate() {
  std::string str_init = "Init";
}

std::string ModelGen::TensorTypeStr(TensorType type) {
  switch (type) {
    case TensorType::FLOAT32:
      return "ANEURALNETWORKS_TENSOR_FLOAT32";
      break;

    case TensorType::INT32:
      return "ANEURALNETWORKS_TENSOR_INT32";
      break;

    case TensorType::UINT8:
      return "ANEURALNETWORKS_TENSOR_QUANT8_ASYMM";
      break;
  }
}

std::string ModelGen::GenerateTensorsCode() {
  Graph& graph = model_.graph();
  std::stringstream ss;

  int count = 0;
  for (const auto& tensor: graph.Tensors()) {
    ss << "ANeuralNetworksOperandType operand_type{\n";

    std::string str_tensor_type;
    switch (tensor.tensor_type()) {
      case TensorType::FLOAT32:
        str_tensor_type = "ANEURALNETWORKS_TENSOR_FLOAT32";
        break;

      case TensorType::INT32:
        str_tensor_type = "ANEURALNETWORKS_TENSOR_INT32";
        break;

      case TensorType::UINT8:
        str_tensor_type = "ANEURALNETWORKS_TENSOR_QUANT8_ASYMM";
        break;
    }
  }
}

void CppGen::GenFiles(const std::vector<std::string>& namespace_vec,
    const boost::filesystem::path& path) {
  std::ofstream tensors_file(path.string() + "/tensors_params.h");

  if (!tensors_file.is_open()) {
    // TODO: throw an exception
    return;
  }

  TensorsHeader tensor_header(model_);
  tensors_file << tensor_header.Assembler(namespace_vec);
  tensors_file.close();
}

}
