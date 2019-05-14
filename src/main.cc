#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>

#include "model.h"
#include "cpp-gen.h"
#include "dump.h"
#include "exception.h"

void GenerateJniFiles(const std::string& str_model, const std::string& str_path, const std::string& java_package)
{
  nnt::Model model(str_model);
  nnt::CppGen cpp(model);
  boost::filesystem::path path(str_path);
  cpp.GenFiles(path, java_package);
  std::cout << "Finish!\n";
}

void GenerateDotFile(const std::string& filename, const std::string& str_model)
{
  nnt::Model model(str_model);
  nnt::DumpGraph dump(model);

  std::ofstream dot_file(filename, std::ofstream::out);

  if (!dot_file.is_open()) {
    std::cerr << "Fail on create dot file: '" << filename << "'\n";
    return;
  }

  std::string dot_src = dump.Dot();
  dot_file.write(dot_src.c_str(), dot_src.length());
  dot_file.close();

  std::cout << "Dot file: '" << filename << "' generated.\n";
}

void PrintInfo(const std::string& str_model)
{
  nnt::Model model(str_model);
  nnt::DumpGraph dump(model);
  std::cout << dump.Info();
}

void PrintDump(const std::string& str_model)
{
  nnt::Model model(str_model);
  nnt::DumpGraph dump(model);
  std::cout << dump.Tensors() << dump.Operators();
}

void PrintTensors(const std::string& str_model)
{
  nnt::Model model(str_model);
  nnt::DumpGraph dump(model);
  std::cout << dump.Tensors();
}

void PrintOperators(const std::string& str_model)
{
  nnt::Model model(str_model);
  nnt::DumpGraph dump(model);
  std::cout << dump.Operators();
}

void PrintWeights(const std::string& str_model)
{
  nnt::Model model(str_model);
  nnt::DumpGraph dump(model);
  std::cout << dump.Weights();
}

int main(int argc, char **argv)
{
  namespace po = boost::program_options;
  std::string str_path;
  std::string java_package;
  std::string str_model;
  std::string str_dot;
  bool flag_info;
  bool flag_operators;
  bool flag_tensors;
  bool flag_weights;

  try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("graph,g", po::value<std::string>(), "generate dot file")
      ("help,h", "show this help screen")
      ("info,i", po::bool_switch(&flag_info), "print high-level info about input/output of the model")
      ("javapackage,j", po::value<std::string>(), "Java package for JNI")
      ("model,m", po::value<std::string>(), "path to flatbuffer model")
      ("operators,o", po::bool_switch(&flag_operators), "print info about operators of the model")
      ("path,p", po::value<std::string>(), "path in which to save output files [default: .]")
      ("tensors,t", po::bool_switch(&flag_tensors), "print info about tensors of the model")
      ("weights,w", po::bool_switch(&flag_weights), "print weights of the model")
      ;

    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0) {
      std::cout << desc << '\n';
      return 0;
    }

    if (vm.count("model") < 1) {
      std::cerr << "--model is required" << '\n';
      std::cerr << desc << '\n';
      return 0;
    }

    str_model = vm["model"].as<std::string>();

    if (flag_info) {
      PrintInfo(str_model);
    }

    if (flag_operators) {
      PrintOperators(str_model);
    }

    if (flag_tensors) {
      PrintTensors(str_model);
    }

    if (flag_weights) {
      PrintWeights(str_model);
    }

    if (vm.count("graph") != 0) {
      str_dot = vm["graph"].as<std::string>();
      GenerateDotFile(str_dot, str_model);
      return 0;
    }

    if (vm.count("path") != 0) {
      str_path = vm["path"].as<std::string>();
    } else {
      str_path = "./";
    }

    if (vm.count("javapackage") != 0) {
      java_package = vm["javapackage"].as<std::string>();
      GenerateJniFiles(str_model, str_path, java_package);
    }

    return 0;

  } catch (const boost::program_options::error &e) {
    std::cerr << "Error: " << e.what() << '\n';
  } catch (const nnt::Exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
  }
}
