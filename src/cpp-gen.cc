#include "cpp-gen.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include "exception.h"

namespace annc {

std::string TensorsHeader::Generate() {
  const std::vector<Buffer>& buffers = model_.Buffers();
  std::string str_buf;

  for (const auto& buf : buffers) {
    if (buf.Data().size() > 0) {
      std::string str_vec = "";
      for (const auto& c : buf.Data()) {
        str_buf += c;
      }
    }
  }

  return str_buf;
}

std::string TensorsHeader::Assembler() {
  return Generate();
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

    default:
      FATAL("Tensor type not valid for Android NNAPI")
  }
}

std::string ModelGen::TensorCppTypeStr(TensorType type) {
  switch (type) {
    case TensorType::FLOAT32:
      return "FLOAT16";
      break;

    case TensorType::INT32:
      return "int32_t";
      break;

    case TensorType::UINT8:
      return "char";
      break;

    default:
      FATAL("Tensor type not valid for Android NNAPI")
  }
}

std::string ModelGen::TensorDim(const std::vector<int>& dim) {
  std::string str_out = "{";

  for (const auto& e : dim) {
    str_out += std::to_string(e) + ",";
  }

  str_out = str_out.substr(0, str_out.length() - 2);
  str_out += "}";

  return str_out;
}

float ModelGen::TensorQuantizationScale(const QuantizationParameters& q) {
  if (q.scale.size() > 0) {
    return q.scale[0];
  } else {
    return 0.0f;
  }
}

int ModelGen::TensorQuantizationZeroPoint(
    const QuantizationParameters& q) {
  if (q.zero_point.size() > 0) {
    return q.zero_point[0];
  } else {
    return 0;
  }
}

std::string ModelGen::GenerateTensorType(const Tensor& tensor, int count) {
  std::stringstream ss;
  std::string dimensions = TensorDim(tensor.shape());

  ss << "uint32_t dimensions_" << count << "[] = " << dimensions << ";\n";
  ss << "ANeuralNetworksOperandType operand_type_" << count << " {\n";

  std::string str_tensor_type = TensorTypeStr(tensor.tensor_type());
  int dimension_count = tensor.shape().size();

  float scale;
  int zero_point;

  if (tensor.HasQuantization()) {
    scale = TensorQuantizationScale(tensor.quantization());
    zero_point = TensorQuantizationZeroPoint(tensor.quantization());
  } else {
    scale = 0.0f;
    zero_point = 0;
  }

  ss << "  .type = " << str_tensor_type << ",\n";
  ss << "  .dimensionCount = " << dimension_count << ",\n";
  ss << "  .dimensions = " << "dimensions_" << count << ",\n";
  ss << "  .scale = " << scale << "f,\n";
  ss << "  .zeroPoint = " << zero_point << "\n";
  ss << "}\n\n";

  return ss.str();
}

std::string ModelGen::CheckStatus(const boost::format& msg) {
  std::stringstream ss;

  // nnapi always check the result of operation
  ss << "if (status != ANEURALNETWORKS_NO_ERROR) {\n";
  ss << "  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n";
  ss << "      \"" << boost::str(msg) << "\");\n";
  ss << "  return false;\n";
  ss << "}\n\n";

  return ss.str();
}

std::string ModelGen::GenerateTensorsCode() {
  Graph& graph = model_.graph();
  std::stringstream ss;

  int count = 0;
  for (const auto& tensor: graph.Tensors()) {
    // insert operand type
    ss << GenerateTensorType(tensor, count);

    // insert nnapi operand
    ss << "status = ANeuralNetworksModel_addOperand(model,";
    ss << "operand_type_" << count << ");\n";
    ss << CheckStatus(boost::format("ANeuralNetworksModel_addOperand failed"
        "for operand %1%")%count);

    size_t buf_size = tensor.buffer().Data().size();

    if (buf_size > 0) {
      // get tensor size
      ss << "tensor_size = " << buf_size << ";\n";

      // insert operand value
      ss << "status = ANeuralNetworksModel_setOperandValueFromMemory(model, ";
      ss << count << ", memory_model, offset, tensor_size);\n\n";
      ss << CheckStatus(boost::format(
          "ANeuralNetworksModel_setOperandValueFromMemory "
          "failed for operand %1%")%count);

      // calculates the offset
      ss << "offset += tensor_size;\n";
    }

    ++count;
  }

  tensor_pos_ = graph.Tensors().size();

  return ss.str();
}

std::string ModelGen::GenerateOpInputs(const std::vector<int>& inputs,
    size_t num_params) {
  // inputs loop
  std::string str_in = "";

  // insert data params like conv filters params
  for (const auto& in_value : inputs) {
    str_in += " " + std::to_string(in_value) + ",";
  }

  // insert hiperparams like conv stride
  size_t tensor_start_pos = tensor_pos_;
  for (; tensor_pos_ < (tensor_start_pos + num_params); tensor_pos_++) {
    str_in += " " + std::to_string(tensor_pos_) + ",";
  }

  str_in = str_in.substr(0, str_in.length() - 1);
  return str_in;
}

std::string ModelGen::GenerateOpOutputs(const std::vector<int>& outputs) {
  // outputs loop
  std::string str_out = "";

  for (const auto& out_value : outputs) {
    str_out += " " + std::to_string(out_value) + ",";
  }

  str_out = str_out.substr(0, str_out.length() - 1);
  return str_out;
}

std::string ModelGen::OpTypeStr(BuiltinOperator op_type) {
  switch (op_type) {
    case BuiltinOperator::ADD:
      return "ANEURALNETWORKS_ADD";
      break;

    case BuiltinOperator::AVERAGE_POOL_2D:
      return "ANEURALNETWORKS_AVERAGE_POOL_2D";
      break;

    case BuiltinOperator::MAX_POOL_2D:
      return "ANEURALNETWORKS_MAX_POOL_2D";
      break;

    case BuiltinOperator::L2_POOL_2D:
      return "ANEURALNETWORKS_L2_POOL_2D";
      break;

    case BuiltinOperator::CONV_2D:
      return "ANEURALNETWORKS_CONV_2D";
      break;

    case BuiltinOperator::RELU:
      return "ANEURALNETWORKS_RELU";
      break;

    case BuiltinOperator::RELU6:
      return "ANEURALNETWORKS_RELU6";
      break;

    case BuiltinOperator::TANH:
      return "ANEURALNETWORKS_TANH";
      break;

    case BuiltinOperator::LOGISTIC:
      return "ANEURALNETWORKS_LOGISTIC";
      break;

    case BuiltinOperator::DEPTHWISE_CONV_2D:
      return "ANEURALNETWORKS_DEPTHWISE_CONV_2D";
      break;

    case BuiltinOperator::CONCATENATION:
      return "ANEURALNETWORKS_CONCATENATION";
      break;

    case BuiltinOperator::SOFTMAX:
      return "ANEURALNETWORKS_SOFTMAX";
      break;

    case BuiltinOperator::FULLY_CONNECTED:
      return "ANEURALNETWORKS_FULLY_CONNECTED";
      break;

    case BuiltinOperator::RESHAPE:
      return "ANEURALNETWORKS_RESHAPE";
      break;

    case BuiltinOperator::SPACE_TO_DEPTH:
      return "ANEURALNETWORKS_SPACE_TO_DEPTH";
      break;

    case BuiltinOperator::LSTM:
      return "ANEURALNETWORKS_LSTM";
      break;

    default:
      FATAL(boost::format("Not supported type on NNAPI"))
  }
}

std::tuple<size_t, std::string> ModelGen::OpParams(const Operator& op) {
  std::stringstream ss;
  size_t num_params = 0;

  auto check = [&op](BuiltinOptionsType type) {
    if (op.builtin_op().type != type) {
      FATAL(boost::format("Operator node type wrong"));
    }
  };

  switch (op.op_code().builtin_code) {
    case BuiltinOperator::ADD:
      ss << "AddScalarInt32(0);\n";
      num_params = 1;
      break;

    case BuiltinOperator::L2_POOL_2D:
    case BuiltinOperator::MAX_POOL_2D:
    case BuiltinOperator::AVERAGE_POOL_2D: {
      check(BuiltinOptionsType::Pool2DOptions);
      const Pool2DOptions& pool_options = static_cast<const Pool2DOptions&>(
          op.builtin_op());

      ss << "AddScalarInt32(" << static_cast<int>(pool_options.padding);
      ss << ");\n";
      ss << "AddScalarInt32(" << pool_options.stride_w << ");\n";
      ss << "AddScalarInt32(" << pool_options.stride_h << ");\n";
      ss << "AddScalarInt32(" << pool_options.filter_width << ");\n";
      ss << "AddScalarInt32(" << pool_options.filter_height << ");\n";
      ss << "AddScalarInt32(";
      ss << static_cast<int>(pool_options.fused_activation_function);
      ss << ");\n";
      num_params = 6;
      break;
    }

    case BuiltinOperator::CONV_2D: {
      check(BuiltinOptionsType::Conv2DOptions);
      const Conv2DOptions& conv_options = static_cast<const Conv2DOptions&>(
          op.builtin_op());

      ss << "AddScalarInt32(" << static_cast<int>(conv_options.padding);
      ss << ");\n";
      ss << "AddScalarInt32(" << conv_options.stride_w << ");\n";
      ss << "AddScalarInt32(" << conv_options.stride_h << ");\n";
      ss << "AddScalarInt32(";
      ss << static_cast<int>(conv_options.fused_activation_function);
      ss << ");\n";
      num_params = 4;
      break;
    }

    case BuiltinOperator::DEPTHWISE_CONV_2D: {
      check(BuiltinOptionsType::DepthwiseConv2DOptions);
      const DepthwiseConv2DOptions& dept_conv_options =
          static_cast<const DepthwiseConv2DOptions&>(op.builtin_op());

      ss << "AddScalarInt32(" << static_cast<int>(dept_conv_options.padding);
      ss << ");\n";
      ss << "AddScalarInt32(" << dept_conv_options.stride_w << ");\n";
      ss << "AddScalarInt32(" << dept_conv_options.stride_h << ");\n";
      ss << "AddScalarInt32(" << dept_conv_options.depth_multiplier << ");\n";
      ss << "AddScalarInt32(";
      ss << static_cast<int>(dept_conv_options.fused_activation_function);
      ss << ");\n";
      num_params = 5;
      break;
    }

    case BuiltinOperator::FULLY_CONNECTED: {
      check(BuiltinOptionsType::FullyConnectedOptions);
      const FullyConnectedOptions& fully_con_options =
          static_cast<const FullyConnectedOptions&>(op.builtin_op());

      ss << "AddScalarInt32(";
      ss << static_cast<int>(fully_con_options.fused_activation_function);
      ss << ");\n";
      num_params = 1;
      break;
    }

    case BuiltinOperator::CONCATENATION: {
      check(BuiltinOptionsType::ConcatenationOptions);
      const ConcatenationOptions& concat_options =
          static_cast<const ConcatenationOptions&>(op.builtin_op());

      ss << "AddScalarInt32(" << concat_options.axis << ");\n";
      ss << "AddScalarInt32(";
      ss << static_cast<int>(concat_options.fused_activation_function);
      ss << ");\n";
      num_params = 2;
      break;
    }

    case BuiltinOperator::SOFTMAX: {
      check(BuiltinOptionsType::SoftmaxOptions);
      const SoftmaxOptions& softmax_options =
          static_cast<const SoftmaxOptions&>(op.builtin_op());

      ss << "AddScalarFloat32(" << softmax_options.beta << ");\n";
      num_params = 1;
      break;
    }

    case BuiltinOperator::SPACE_TO_DEPTH: {
      check(BuiltinOptionsType::SpaceToDepthOptions);
      const SpaceToDepthOptions& space2depth_options =
          static_cast<const SpaceToDepthOptions&>(op.builtin_op());

      ss << "AddScalarInt32(" << space2depth_options.block_size << ");\n";
      num_params = 1;
      break;
    }

    case BuiltinOperator::LSTM: {
      check(BuiltinOptionsType::LSTMOptions);
      // TODO: Check better on TfLite how lstm parametes is filled
      const LSTMOptions& lstm_options = static_cast<const LSTMOptions&>(
          op.builtin_op());

      ss << "AddScalarInt32(";
      ss << static_cast<int>(lstm_options.fused_activation_function);
      ss << ");\n";
      ss << "AddScalarFloat32(" << lstm_options.cell_clip << ");\n";
      ss << "AddScalarFloat32(" << lstm_options.proj_clip << ");\n";
      num_params = 3;
      break;
    }

    default:
      num_params = 0;
  }

  return std::tuple<size_t, std::string>(num_params, ss.str());
}

std::string ModelGen::GenerateOpCode() {
  Graph& graph = model_.graph();
  std::stringstream ss;

  int count = 0;
  for (const auto& op: graph.Operators()) {
    size_t num_params;
    std::string str_params;
    std::tie(num_params, str_params) = OpParams(op);
    ss << str_params << "\n";
    ss << "uint32_t input_operands_" << count << "[";
    ss << op.inputs().size() <<"] = { ";
    ss << GenerateOpInputs(op.inputs(), num_params) << " };\n";

    ss << "uint32_t output_operands_" << count << "[";
    ss << op.outputs().size() <<"] = {";
    ss << GenerateOpOutputs(op.outputs()) << " };\n\n";

    ss << "status = ANeuralNetworksModel_addOperation(model, ";
    ss << OpTypeStr(op.op_code().builtin_code) << ", sizeof(input_operands_" ;
    ss << count <<"), input_operands_" << count << ", ";
    ss << "sizeof(output_operands_" << count << "), ";
    ss << "output_operands_" << count << ");\n";

    ss << CheckStatus(boost::format(
        "ANeuralNetworksModel_addOperation failed for operation %1%")%count);

    ++count;
  }

  return ss.str();
}

std::string ModelGen::Assembler() {
  std::string code;
  code = GenerateTensorsCode();
  code += GenerateOpCode();

  return code;
}

void CppGen::GenFiles(const std::vector<std::string>& namespace_vec,
    const boost::filesystem::path& path) {
  GenTensorsDataFile(path);
  GenCppFile(path);
}

void CppGen::GenTensorsDataFile(const boost::filesystem::path& path) {
  std::ofstream tensors_file(path.string() + "/weights_biases.bin",
      std::ofstream::out | std::ofstream::binary);

  if (!tensors_file.is_open()) {
    FATAL("Fail on create tensors_params.h file")
  }

  TensorsHeader tensor_header(model_);
  std::string buf = tensor_header.Assembler();
  tensors_file.write(buf.c_str(), buf.length());
  tensors_file.close();
}

void CppGen::GenCppFile(const boost::filesystem::path& path) {
  std::ofstream cc_file(path.string() + "/nn.cc",
      std::ofstream::out | std::ofstream::binary);

  if (!cc_file.is_open()) {
    FATAL("Fail on create nn.cc file")
  }

  ModelGen model(model_);
  std::string code = model.Assembler();
  cc_file.write(code.c_str(), code.length());
  cc_file.close();
}

}
