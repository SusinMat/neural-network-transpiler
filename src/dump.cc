#include <iostream>
#include <sstream>

#include "dump.h"

namespace nnt {

  template<class T>
    std::string VectorToStr(const std::vector<T>& vec) {
      std::stringstream ss;

      ss << "[";
      for (const auto&i : vec) {
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


  std::string DumpGraph::Weights() {
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

      ss << "\n └─input shapes:";
      for (const auto& input : op.inputs()) {
        ss << "  " << input << " : " << VectorToStr(graph.Tensors()[input].shape());
      }
      ss << "\n";

    }

    return ss.str();


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
