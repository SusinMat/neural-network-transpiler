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

void Info(const std::string& str_model)
{
  nnt::Model model(str_model);
  nnt::DumpGraph dump(model);
  std::cout << dump.Info();
}

void PrintDump(const std::string& str_model)
{
  nnt::Model model(str_model);
  nnt::DumpGraph dump(model);
  dump.Print();
  std::cout << std::endl;
}

int main(int argc, char **argv)
{
  namespace po = boost::program_options;
  std::string str_path;
  std::string java_package;
  std::string str_model;
  std::string str_dot;
  bool flag_info;
  bool flag_dump;

  try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("dump,d", po::bool_switch(&flag_dump), "print info about tensors and operators of the model")
      ("graph,g", po::value<std::string>(), "generate dot file")
      ("help,h", "Help screen")
      ("info,i", po::bool_switch(&flag_info), "print info about input/output of the model")
      ("javapackage,j", po::value<std::string>(), "Java package for JNI")
      ("model,m", po::value<std::string>(), "path to flatbuffer model")
      ("path,p", po::value<std::string>(), "path in which to save output files [default: .]");

    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0) {
      std::cout << desc << '\n';
      return 0;
    }

    if (vm.count("model") < 1) {
      std::cerr << "--model requires an argument" << '\n';
      std::cerr << desc << '\n';
      return 0;
    }

    str_model = vm["model"].as<std::string>();

    if (flag_info) {
      Info(str_model);
    }

    if (flag_dump) {
      PrintDump(str_model);
    }

    if (vm.count("dot") != 0) {
      str_dot = vm["dot"].as<std::string>();
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
