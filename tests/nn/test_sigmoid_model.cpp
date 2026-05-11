
#include <iostream>

#include <gtest/gtest.h>

#include "nn/activation.h"
#include "nn/layer.h"
#include "nn/loss.h"


using namespace NN;

TEST(ModelTests, SequentialSigmoidModel) {
  auto sigmoid = Activations::Sigmoid::create();
  auto mae = Loss::MAE::create();

  auto model = Layers::Sequential({
    Layers::Linear::create(3, 3, sigmoid, true),
    Layers::Linear::create(3, 2, sigmoid, true),
    Layers::Linear::create(2, 1, sigmoid, true)
  });

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

  Eigen::MatrixXd x_input_test(2, 3);
  x_input_test
    <<
      1, 1, 1,
      0, 0, 0;

  Eigen::MatrixXd y_targ_test(2, 1);
  y_targ_test
    <<
      1,
      0;

  std::cout
    << "\n=====Train Data====="
    << "\nInput train:\n"
    << x_input_train
    << "\nTarget train:\n"
    << y_targ_train
    << "\n=====Test Data====="
    << "\nInput test:\n"
    << x_input_test
    << "\nTarget test:\n"
    << y_targ_test
    << std::endl;


  double loss = 1;
  double rate = 1;
  std::cout
    << "\nLearning rate: " << rate << std::endl;

  for (size_t i = 0; i < 10000; i++) {
    model.forward(x_input_train);
    model.backward(model.getResult() - y_targ_train);
    model.findBiases(rate);

    model.forward(x_input_train);
    model.backward(model.getResult() - y_targ_train);
    model.findWeights(rate);

    if (i % 1000 == 0) {
      model.forward(x_input_test);
      loss = mae->function(model.getResult(), y_targ_test);
      std::cout 
        << "Epoch: " << i << "/10000\t"
        << "Loss: " << loss
        << std::endl;
    }
  }

  EXPECT_NEAR( loss, 0, 0.1 );
}

int main (int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
