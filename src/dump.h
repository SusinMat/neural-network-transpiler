#ifndef DROIDNET_DUMP_H
#define DROIDNET_DUMP_H

#include <string>
#include <vector>

#include "model.h"

namespace annc {

class DumpGraph {
 public:
  DumpGraph(Model& model): model_(model) {}

  void Print();

  std::string Dot();

 private:
  Model& model_;
};

}

#endif
