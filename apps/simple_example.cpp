
#include <fstream>
#include <iostream>

#include "nn/activation.h"
#include "nn/layer.h"
#include "nn/loss.h"


int main (int argc, char *argv[]) {

  // create activation and loss function
  auto sigmoid = Activations::Sigmoid::create();
  auto mae = Loss::MAE::create();
  
  // Create Model
  auto model = Layers::Sequential({
    Layers::Linear::create(3, 3, sigmoid, true),
    Layers::Linear::create(3, 2, sigmoid, true),
    Layers::Linear::create(2, 1, sigmoid, true)
  });

  Eigen::MatrixXd in = Eigen::MatrixXd::Random(1, 3);
  Eigen::MatrixXd err = Eigen::MatrixXd::Random(1, 3);

  // 
  Eigen::MatrixXd x_input_train(4, 3);
  x_input_train
    <<
      0, 1, 0,
      0, 1, 1,
      1, 0, 0,
      0, 0, 1;

  Eigen::MatrixXd y_targ_train(4, 1);
  y_targ_train
    <<
      1,
      1,
      0,
      0;
  
  for (size_t i = 0; i < 1000; i++) {
    model.forward(x_input_train);
    model.backward(model.getResult() - y_targ_train);
    model.findBiases(1);

    model.forward(x_input_train);
    model.backward(model.getResult() - y_targ_train);
    model.findWeights(1);

    if (i % 200 == 0) {
      std::cout << mae->function(model.getResult(), y_targ_train) << std::endl;
      std::cout << "--------------" << std::endl;
    }
  }

  for (size_t i = 0; i < 10000; i++) {
    model.forward(x_input_train);
    model.backward(model.getResult() - y_targ_train);
    model.findBiases(0.1);

    model.forward(x_input_train);
    model.backward(model.getResult() - y_targ_train);
    model.findWeights(0.1);

    if (i % 500 == 0) {
      std::cout << mae->function(model.getResult(), y_targ_train) << std::endl;
      std::cout << "--------------" << std::endl;
    }
  }

  // Saving model
  {
    std::ofstream ofs("model.dat");
    boost::archive::text_oarchive oa(ofs);
    oa << model;
  }

  // Loading model
  Layers::Sequential model_loaded;
  {
    std::ifstream ifs("model.dat");
    boost::archive::text_iarchive ia(ifs);
    ia >> model_loaded;
  }

  model_loaded.forward(x_input_train);
  std::cout << "RESULT:"<< std::endl;
  std::cout << model_loaded.getResult() << std::endl;
  std::cout << "LOSS:"<< std::endl;
  std::cout << mae->function(model.getResult(), y_targ_train) << std::endl;
  return 0;
}
