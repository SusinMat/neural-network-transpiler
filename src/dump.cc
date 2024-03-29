#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "dump.h"

namespace nnt {

  template<class T>std::string VectorToStrWithShape(const std::vector<T>& vec, const std::vector<int>& shape)
  {
    std::string str;

    switch (shape.size()) {
    case 0:
      str = "Empty shape vector";
      break;
    case 1:
      if (vec.size() == 0) {
        assert(shape[0] == 1);
        str = "[ ]";
        break;
      }
      assert(vec.size() == (size_t)(shape[0]));

      str += "[ ";

      for (int i = 0; i < shape[0]; ++i) {
        std::stringstream ss;
        ss << std::setprecision(std::numeric_limits<T>::max_digits10);
        ss << vec[i];
        str += ss.str() + ", ";
      }
      str = str.substr(0, str.length() - 2);

      str += " ]";
      break;
    case 2:
      if (vec.size() == 0) {
        str = "Empty data vector";
        break;
      }
      assert(vec.size() == (size_t)(shape[0] * shape[1]));

      str += "[ ";

      for (int i = 0; i < shape[0]; ++i) {
        str += "[ ";
        for (int j = 0; j < shape[1]; ++j) {
          std::stringstream ss;
          ss << std::setprecision(std::numeric_limits<T>::max_digits10);
          ss << vec[i * (shape[1])
                  + j];
          str += ss.str() + ", ";
        }
        str = str.substr(0, str.length() - 2);
        str += "], ";
      }
      str = str.substr(0, str.length() - 2);
      str += " ]";
      break;
    case 3:
      if (vec.size() == 0) {
        str = "Empty data vector";
        break;
      }
      assert(vec.size() == (size_t)(shape[0] * shape[1] * shape[2]));

      str += "[ ";

      for (int i = 0; i < shape[0]; ++i) {
        str += "[ ";
        for (int j = 0; j < shape[1]; ++j) {
          str += "[ ";
          for (int k = 0; k < shape[2]; ++k) {
            std::stringstream ss;
            ss << std::setprecision(std::numeric_limits<T>::max_digits10);
            ss << vec[i * (shape[1] * shape[2])
                    + j * (shape[2])
                    + k];
            str += ss.str() + ", ";
          }
          str = str.substr(0, str.length() - 2);
          str += "], ";
        }
        str = str.substr(0, str.length() - 2);
        str += " ], ";
      }
      str = str.substr(0, str.length() - 2);
      str += "]";
      break;
    case 4:
      if (vec.size() == 0) {
        str = "Empty data vector";
        break;
      }

      assert(vec.size() == (size_t)(shape[0] * shape[1] * shape[2] * shape[3]));

      str += "[ ";

      for (int i = 0; i < shape[0]; ++i) {
        str += "[ ";
        for (int j = 0; j < shape[1]; ++j) {
          str += "[ ";
          for (int k = 0; k < shape[2]; ++k) {
            str += "[ ";
            for (int l = 0; l < shape[3]; ++l) {
              std::stringstream ss;
              ss << std::setprecision(std::numeric_limits<T>::max_digits10);
              ss << vec[i * (shape[1] * shape[2] * shape[3])
                      + j * (shape[2] * shape[3])
                      + k * (shape[3])
                      + l];
              str += ss.str() + ", ";
            }
            str = str.substr(0, str.length() - 2);
            str += " ], ";
          }
          str = str.substr(0, str.length() - 2);
          str += " ], ";
        }
        str = str.substr(0, str.length() - 2);
        str += " ], ";
      }
      str = str.substr(0, str.length() - 2);
      str += " ]";
      break;
    default:
      str = "Shape vector is not a [0-4]D vector.";
    }

    str += "\n";

    return str;
  }

  template<class T>std::string VectorToStr(const std::vector<T>& vec)
  {
    std::stringstream ss;

    if (vec.size() == 0) {
      ss << "[ ]";
      return ss.str();
    }

    ss << "[" ;
    for (const auto& i : vec) {
      ss << i << ", ";
    }

    std::string str = ss.str();
    str = str.substr(0, str.length() - 2);

    str += "]";

    return str;
  }

  std::string DumpGraph::Tensors() {
    std::stringstream ss;
    Graph& graph = model_.graph();

    ss << "\nTensors:\n";
    int count = 0;
    for (const auto& tensor: graph.Tensors()) {
      ss << "[" << count++ << "] ";
      ss << "name: " << tensor.name() << " [ ";
      for (const auto&i : tensor.shape()) {
        ss << i << " ";
      }
      ss << "] ";
      ss << "buffer: " << tensor.buffer_index() << "\n";
    }

    ss << "\n";
    return ss.str();
  }


  std::vector<float> ByteVectorToFloatVector(std::vector<unsigned char> bytes)
  {
    std::vector<float> floats;

    union {unsigned char as_bytes[4]; float as_float;} helper_union;

    assert(bytes.size() % 4 == 0);
    floats.reserve(bytes.size() / 4);

    for (size_t i = 0; i < bytes.size(); i += 4) {
      float float_copy = 0.0f;
      for (int j = 0; j < 4; j++)
        helper_union.as_bytes[j] = bytes[i + j];
      float_copy = helper_union.as_float;
      floats.push_back(float_copy);
    }

    return floats;
  }

  std::vector<int32_t> ByteVectorToInt32Vector(std::vector<unsigned char> bytes)
  {
    std::vector<int32_t> int32s;

    union {unsigned char as_bytes[4]; int32_t as_int32;} helper_union;

    assert(bytes.size() % 4 == 0);
    int32s.reserve(bytes.size() / 4);

    for (size_t i = 0; i < bytes.size(); i += 4) {
      int32_t int32_copy = 0.0f;
      for (int j = 0; j < 4; j++)
        helper_union.as_bytes[j] = bytes[i + j];
      int32_copy = helper_union.as_int32;
      int32s.push_back(int32_copy);
    }

    return int32s;
  }

  std::vector<uint16_t> ByteVectorToUInt8Vector(std::vector<unsigned char> bytes)
  {
    std::vector<uint16_t> uint16s;

    uint16s.reserve(bytes.size());

    for (size_t i = 0; i < bytes.size(); ++i) {
      uint16s.push_back(bytes[i]);
    }

    return uint16s;
  }

  std::string PaddingEnumToString(Padding padding)
  {
    std::string str = "";
    switch(padding) {
      case Padding::UNKNOWN:
        str = "UNKNOWN";
        break;
      case Padding::SAME:
        str = "SAME";
        break;
      case Padding::VALID:
        str = "VALID";
        break;
      default:
        std::cout << "Error: unkown padding: " << (int)padding << "\n";
        exit(EXIT_FAILURE);
        break;
    }
    return str;
  }

  std::string ActivationFunctionEnumToString(ActivationFunctionType activation_function)
  {
    std::string str = "";
    switch(activation_function) {
      case ActivationFunctionType::NONE:
        str = "NONE";
        break;
      case ActivationFunctionType::RELU:
        str = "RELU";
        break;
      case ActivationFunctionType::RELU1:
        str = "RELU1";
        break;
      case ActivationFunctionType::RELU6:
        str = "RELU6";
        break;
      case ActivationFunctionType::TANH:
        str = "TANH";
        break;
      case ActivationFunctionType::SIGN_BIT:
        str = "SIGN_BIT";
        break;
      default:
        std::cout << "Error: unkown activation_function: " << (int)activation_function << "\n";
        exit(EXIT_FAILURE);
        break;
    }
    return str;
  }

  std::string QuantizationParamsToStr(const Tensor &tensor)
  {
    std::stringstream ss;
    const QuantizationParameters &quantization = tensor.quantization();
    const std::streamsize default_stringstream_precision = ss.precision();
    const float quantization_min = (quantization.min.size() > 0) ? quantization.min[0] : NAN;
    const float quantization_max = (quantization.max.size() > 0) ? quantization.max[0] : NAN;

    ss << std::setprecision(std::numeric_limits<float>::max_digits10);
    ss << "[" << quantization_min;
    ss << ", " << quantization_max;
    ss << ", "   << quantization.scale[0];
    ss <<", "    << quantization.zero_point[0] << "]";
    ss << std::setprecision(default_stringstream_precision);

    return ss.str();
  }

  std::string DumpGraph::Weights()
  {
    std::stringstream ss;
    Graph& graph = model_.graph();

    ss << "\nOperators:\n";
    for (const auto &op: graph.Operators()) {
      const BuiltinOptions& options = op.builtin_op();
      const BuiltinOptionsType options_type = options.type;

      ss << "builtin_op: " << op.builtin_op_str() << "(" << op.index()<< "), ";
      // std::cout << "builtin_op: " << op.builtin_op_str() << "(" << op.index()<< "), " << "\n";
      ss << "options: { ";
      switch(options_type) {
        case BuiltinOptionsType::Conv2DOptions:
        {
          const Conv2DOptions &downcast_opt = static_cast<const Conv2DOptions &>(options);
          ss << "padding:" << PaddingEnumToString(downcast_opt.padding) << "(" << (int)downcast_opt.padding << ")" << ", ";
          ss << "stride_w:" << downcast_opt.stride_w << ", ";
          ss << "stride_h:" << downcast_opt.stride_h << ", ";
#ifdef NEWER_TENSORFLOW
          ss << "dilation_w_factor:" << downcast_opt.dilation_stride_w << ", ";
          ss << "dilation_h_factor:" << downcast_opt.dilation_stride_h <<", ";
#endif
          ss << "fused_activation_function:" << ActivationFunctionEnumToString(downcast_opt.fused_activation_function) << "(" << (int)downcast_opt.fused_activation_function << ")" << ", ";
          break;
        }
        case BuiltinOptionsType::DepthwiseConv2DOptions:
        {
          const DepthwiseConv2DOptions &downcast_opt = static_cast<const DepthwiseConv2DOptions &>(options);
          ss << "padding:" << PaddingEnumToString(downcast_opt.padding) << "(" << (int)downcast_opt.padding << ")" << ", ";
          ss << "stride_w:" << downcast_opt.stride_w << ", ";
          ss << "stride_h:" << downcast_opt.stride_h << ", ";
          ss << "depth_multiplier:" << downcast_opt.depth_multiplier << ", ";
          ss << "fused_activation_function:" << ActivationFunctionEnumToString(downcast_opt.fused_activation_function) << "(" << (int)downcast_opt.fused_activation_function << ")" << ", ";
          break;
        }
        case BuiltinOptionsType::Pool2DOptions:
        {
          const Pool2DOptions &downcast_opt = static_cast<const Pool2DOptions &>(options);
          ss << "padding:" << PaddingEnumToString(downcast_opt.padding) << "(" << (int)downcast_opt.padding << ")" << ", ";
          ss << "stride_w:" << downcast_opt.stride_w << ", ";
          ss << "stride_h:" << downcast_opt.stride_h << ", ";
          ss << "filter_width:" << downcast_opt.filter_width << ", ";
          ss << "filter_height:" << downcast_opt.filter_height << ", ";
          ss << "fused_activation_function:" << ActivationFunctionEnumToString(downcast_opt.fused_activation_function) << "(" << (int)downcast_opt.fused_activation_function << ")" << ", ";
          if (op.op_code().builtin_code == BuiltinOperator::AVERAGE_POOL_2D) {
            ss << "pooling_type:" << "\"AVG\"" << ", ";
          } else if (op.op_code().builtin_code == BuiltinOperator::MAX_POOL_2D) {
            ss << "pooling_type:" << "\"MAX\"" << ", ";
          } else {
            std::cout << "Error:" << int(op.op_code().builtin_code) << " is an invalid pooling type" << "\n";
            exit(EXIT_FAILURE);
          }
          break;
        }
        case BuiltinOptionsType::FullyConnectedOptions:
        {
          const FullyConnectedOptions &downcast_opt = static_cast<const FullyConnectedOptions &>(options);
          ss << "fused_activation_function:" << ActivationFunctionEnumToString(downcast_opt.fused_activation_function) << "(" << (int)downcast_opt.fused_activation_function << ")" << ", ";
          break;
        }
        case BuiltinOptionsType::SoftmaxOptions:
        {
          const SoftmaxOptions &downcast_opt = static_cast<const SoftmaxOptions &>(options);
          ss << "beta:" << std::to_string(downcast_opt.beta) << ", ";
          break;
        }
        case BuiltinOptionsType::ConcatenationOptions:
        {
          const ConcatenationOptions &downcast_opt = static_cast<const ConcatenationOptions &>(options);
          ss << "axis:" << downcast_opt.axis << ", ";
          ss << "fused_activation_function:" << ActivationFunctionEnumToString(downcast_opt.fused_activation_function) << "(" << (int)downcast_opt.fused_activation_function << ")" << ", ";
          // std::cout << "axis:" << downcast_opt.axis << ", " << "fused_activation_function:" << ActivationFunctionEnumToString(downcast_opt.fused_activation_function) << "(" << (int)downcast_opt.fused_activation_function << ")" << ", " << "\n";
          break;
        }
        case BuiltinOptionsType::AddOptions:
        {
          const AddOptions &downcast_opt = static_cast<const AddOptions &>(options);
          ss << "fused_activation_function:" << ActivationFunctionEnumToString(downcast_opt.fused_activation_function) << "(" << (int)downcast_opt.fused_activation_function << ")" << ", ";
          break;
        }
        case BuiltinOptionsType::ReshapeOptions:
        {
          const ReshapeOptions &downcast_opt = static_cast<const ReshapeOptions &>(options);
          ss << "new_shape:" << VectorToStr(downcast_opt.new_shape) << ", ";
          break;
        }
        case BuiltinOptionsType::SqueezeOptions:
        {
          const SqueezeOptions &downcast_opt = static_cast<const SqueezeOptions &>(options);
          ss << "squeeze_dims:" << VectorToStr(downcast_opt.squeeze_dims) << ", ";
          break;
        }
        case BuiltinOptionsType::PadOptions:
        {
          const PadOptions &downcast_opt = static_cast<const PadOptions &>(options);
        }
        break;
        case BuiltinOptionsType::MulOptions:
        {
          const MulOptions &downcast_opt = static_cast<const MulOptions &>(options);
        }
        break;
        case BuiltinOptionsType::MeanOptions:
        {
          const MeanOptions &downcast_opt = static_cast<const MeanOptions &>(options);
          ss << "keep_dims:" << std::to_string(downcast_opt.keep_dims) << ", ";
        }
        break;
        default:
          std::cout << "Error: DumpGraph::Weights() function is unprepared to handle the options of this operation: " << op.builtin_op_str() << "\n";
          exit(EXIT_FAILURE);
          break;
      }
      ss << "}, ";
      ss << "inputs:";
      for (const auto &i : op.inputs()) {
        ss << " " << i;
      }
      ss << ", ";

      ss << "outputs:";
      for (const auto &i : op.outputs()) {
        ss << " " << i;
      }

      ss << "\n + input tensors:\n";
      for (const auto &input : op.inputs()) {
        const Tensor &tensor = graph.Tensors()[input];
        std::vector<unsigned char> tensor_data = tensor.buffer().Data();

        ss << "   * " << input << ":s=" << VectorToStr(tensor.shape());
        ss << "," << "t=" << TensorType(tensor);
        if (tensor.HasQuantization()) {
          ss << ",q=";
          ss << QuantizationParamsToStr(tensor);
        }

#if true
        if (tensor_data.size() == 0) {
          ss << "," << "d=" << "\n" << VectorToStrWithShape(tensor_data, tensor.shape());
        } else if (tensor.tensor_type() == TensorType::FLOAT32) {
          std::vector<float> data_array(ByteVectorToFloatVector(tensor_data));
          ss << "," << "d=" << "\n" << VectorToStrWithShape(data_array, tensor.shape());
        } else if (tensor.tensor_type() == TensorType::INT32) {
          std::vector<int32_t> data_array(ByteVectorToInt32Vector(tensor_data));
          ss << "," << "d=" << "\n" << VectorToStrWithShape(data_array, tensor.shape());
        } else if (tensor.tensor_type() == TensorType::UINT8) {
          std::vector<uint16_t> data_array(ByteVectorToUInt8Vector(tensor_data));
          ss << "," << "d=" << "\n" << VectorToStrWithShape(data_array, tensor.shape());
        }
#else
        ss << "\n";
#endif
      }

      ss << "\n + output tensors:\n";
      for (const auto &output : op.outputs()) {
        const Tensor &tensor = graph.Tensors()[output];
        std::vector<unsigned char> tensor_data = tensor.buffer().Data();
        std::vector<float> data_array(ByteVectorToFloatVector(tensor_data));

        ss << "   * " << output << ":s=" << VectorToStr(tensor.shape());
        ss << "," << "t=" << TensorType(tensor);
        if (tensor.HasQuantization()) {
          ss << ",q=";
          ss << QuantizationParamsToStr(tensor);
        }
      }
      ss << "\n";
      ss << "\n";
    }

    return ss.str();
  }


  std::string DumpGraph::Operators() {
    std::stringstream ss;
    Graph& graph = model_.graph();

    ss << "\nOperators:\n";
    for (const auto& op: graph.Operators()) {
      ss << "index: " << op.index() << ", ";
      ss << "builtin_op: " << op.builtin_op_str() << ", ";

      ss << "inputs:";
      for (const auto& i : op.inputs()) {
        ss << " " << i;
      }
      ss << ", ";

      ss << "outputs:";
      for (const auto& i : op.outputs()) {
        ss << " " << i;
      }
      ss << "\n";
    }

    ss << "\n";
    return ss.str();
  }


  std::string DumpGraph::Info() {
    std::stringstream ss;

    Graph& graph = model_.graph();
    const auto& tensors = graph.Tensors();

    ss << "::Inputs::\n";
    for (const auto& i : graph.Inputs()) {
      ss << " " << tensors[i].name() << "<" << TensorType(tensors[i]) << ">"
        << " [" << TensorShape(tensors[i]) << "]";

      if (tensors[i].HasQuantization()) {
        ss << " (quantized)\n";
        const QuantizationParameters& quant = tensors[i].quantization();
        ss << "   └─ Quant: {min:" << VectorToStr(quant.min) << ", max:"
          << VectorToStr(quant.max) << ", scale: " << VectorToStr(quant.scale)
          << ", zero_point:" << VectorToStr(quant.zero_point) << "}\n";
      } else {
        ss << "\n";
      }
    }
    ss << "\n";

    ss << "::Outputs::\n";
    for (const auto& i : graph.Outputs()) {
      ss << " " << tensors[i].name() << "<" << TensorType(tensors[i]) << ">"
        << " [" << TensorShape(tensors[i]) << "]";

      if (tensors[i].HasQuantization()) {
        ss << " (quantized)\n";
      } else {
        ss << "\n";
      }
    }

    ss << "\n";
    return ss.str();
  }


  std::string DumpGraph::Dot() {
    std::stringstream ss;

    Graph& graph = model_.graph();

    ss << "digraph {\n";

    int count = 0;
    for (const auto& tensor: graph.Tensors()) {
      ss << "  T" << count++ << " [";
      ss << "shape=box label=\"" << FormatTensorName(tensor.name());
      ss << " [" << TensorShape(tensor) << "]\"]\n";
    }

    count = 0;
    for (const auto& op: graph.Operators()) {
      ss << "  O" << count++ << " [";
      ss << "label=\"" << op.builtin_op_str() << "\"]\n";
    }

    count = 0;
    for (const auto& op: graph.Operators()) {
      for (const auto& i : op.inputs()) {
        ss << "  T" << i << " -> " << "O" << count << "\n";
      }

      for (const auto& i : op.outputs()) {
        ss << "  O" << count << " -> " << "T" << i << "\n";
      }

      ++count;
    }

    ss << "}\n";

    return ss.str();
  }

  std::string DumpGraph::TensorType(const Tensor& tensor) {
    switch (tensor.tensor_type()) {
      case TensorType::FLOAT32:
        return std::string("FLOAT32");
        break;

      case TensorType::FLOAT16:
        return std::string("FLOAT16");
        break;

      case TensorType::INT32:
        return std::string("INT32");
        break;

      case TensorType::UINT8:
        return std::string("UINT8");
        break;

      case TensorType::INT64:
        return std::string("INT64");
        break;

      case TensorType::STRING:
        return std::string("STRING");
        break;
#ifdef NEWER_TENSORFLOW
      case TensorType::BOOL:
        return std::string("BOOL");
        break;
#endif
    }

    return std::string();
  }

  std::string DumpGraph::FormatTensorName(const std::string& name) {
    size_t pos = name.find_last_of('/');

    if (pos != std::string::npos) {
      return name.substr(pos);
    }

    return name;
  }

  std::string DumpGraph::TensorShape(const Tensor& tensor) {
    std::stringstream ss;

    for (const auto&i : tensor.shape()) {
      ss << i << ", ";
    }

    std::string str = ss.str();
    str = str.substr(0, str.length() - 2);
    return str;
  }

}
