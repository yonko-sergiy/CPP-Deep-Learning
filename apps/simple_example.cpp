
#include <fstream>
#include <iostream>

#include "nn/activation.h"
#include "nn/layer.h"
#include "nn/loss.h"


using namespace NN;

int main (int argc, char *argv[]) {

  // create activation and loss function
  auto sigmoid = Activations::Sigmoid::create();
  auto mae = Loss::MAE::create();
  
  // Create Model
  auto model = Layers::Sequential({
    Layers::Linear::create(4, 3, sigmoid, true),
    Layers::Linear::create(3, 2, sigmoid, true),
    Layers::Linear::create(2, 1, sigmoid, true)
  });

  Eigen::MatrixXd in = Eigen::MatrixXd::Random(1, 3);
  Eigen::MatrixXd err = Eigen::MatrixXd::Random(1, 3);

  Eigen::MatrixXd x_input_train(12, 4);
  x_input_train  
    <<
      0, 0, 0, 0,
      0, 0, 0, 1,
      0, 0, 1, 0,
      0, 0, 1, 1,
      0, 1, 0, 0,
      0, 1, 0, 1,
      1, 0, 1, 0,
      1, 0, 1, 1,
      1, 1, 0, 0,
      1, 1, 0, 1,
      1, 1, 1, 0,
      1, 1, 1, 1;
 
  Eigen::MatrixXd y_targ_train(12, 1);
  y_targ_train
    <<
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0;

  std::cout
    << "=====Train Data=====\n"
    << "Input train:\n"
    << x_input_train
    << "\nTarget train:\n"
    << y_targ_train
    << "\n"
    << std::endl;

  Eigen::MatrixXd x_input_test(4, 4);
  x_input_test
    <<
      0, 1, 1, 0,
      0, 1, 1, 1,
      1, 0, 0, 0,
      1, 0, 0, 1;

  Eigen::MatrixXd y_targ_test(4, 1);
  y_targ_test
    <<
      1,
      1,
      0,
      0;

  std::cout
    << "=====Test Data=====\n"
    << "Input test:\n"
    << x_input_train
    << "\nTarget test:\n"
    << y_targ_train
    << std::endl;

  double rate = 1;
  double loss = 1;
  std::cout << "\nLearning rate: " << rate << std::endl;
  for (size_t i = 0; i < 5000; i++) {
    model.forward(x_input_train);
    model.backward(model.getResult() - y_targ_train);
    model.findBiases(rate);

    model.forward(x_input_train);
    model.backward(model.getResult() - y_targ_train);
    model.findWeights(rate);

    if (i % 500 == 0) {
      model.forward(x_input_test);
      loss = mae->function(model.getResult(), y_targ_test);
      std::cout << "Epoch: " << i << "/5000\t\tLoss: " << loss << std::endl;
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

  model_loaded.forward(x_input_test);
  std::cout << "RESULT:"<< std::endl;
  std::cout << model_loaded.getResult() << std::endl;
  std::cout << "TARGET:"<< std::endl;
  std::cout << y_targ_test << std::endl;
  std::cout << "LOSS:"<< std::endl;
  std::cout << mae->function(model.getResult(), y_targ_train) << std::endl;
  return 0;
}
