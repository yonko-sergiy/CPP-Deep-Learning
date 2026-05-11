
#include <cmath>
#include <gtest/gtest.h>

#include "nn/layer.h"
#include "nn/activation.h"
#include <iostream>


// Tests factorial of 0.
TEST(FactorialTest, HandlesZeroInput) {
  EXPECT_EQ(1, 1);
}

TEST(FactorialTest1, HandlesOneInput) {
  EXPECT_EQ(1, 1);
}

TEST(Other, otr) {
  EXPECT_EQ(3, 3);
}

TEST(Other1, otr2) {
  EXPECT_EQ(3, 3);
}



// #include "nn.hpp"

// Eigen::MatrixXd LossMAE (Eigen::MatrixXd target, Eigen::MatrixXd predict) {
//   Eigen::MatrixXd difference = (target.array() - predict.array());
//   return difference.array() / difference.size();
// }
//
// Layer CreateLayer(SigmoidLayer, 3, 3, true) {
//   return shared_pointer
// };
// auto my_layer = CreateLayer(SigmoidLayer, 3, 3, true) {

int main (int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  // std::cout << "####test#####" << std::endl;
  //
  // auto a = Activations::Activation::create();
  // a = Activations::Sigmoid::create();
  //
  // auto my_layer = Layers::Linear(3, 4, a, false);
  // auto my_layer2 = Layers::Linear::create(4, 2, a, false);
  // auto my_layer3 = Layers::Linear::create(2, 3, a, false);
  // std::cout << "####post test#####" << std::endl;
  //

  // auto my_layer = Layers::Linear(4096, 2048, a, true);
  // auto my_layer2 = Layers::Linear::create(2048, 512, a, true);
  // auto my_layer3 = Layers::Linear::create(512, 4, a, true);
  //
  //
  // Eigen::MatrixXd m = Eigen::MatrixXd::Random(20, 4096);
  // Eigen::MatrixXd b = Eigen::MatrixXd::Random(20, 4);
  // std::shared_ptr<Eigen::MatrixXd> x =
  //   std::make_shared<Eigen::MatrixXd>(m);  // copy into shared_ptr
  //
  // my_layer.setInput(m);
  // my_layer.bindNextLayer(my_layer2);
  // my_layer2->bindNextLayer(my_layer3);
  //
  // my_layer.forward();
  // my_layer2->forward();
  // my_layer3->forward();
  //
  // my_layer3->backward(b);
  // my_layer2->backward();
  // my_layer.backward();
  //
  // my_layer3->findWeights();
  // my_layer2->findWeights();
  // my_layer.findWeights();
  //
  // my_layer3->findBiases();

  // std::cout << my_layer2->getResult() << std::endl;

  // std::weak_ptr<Eigen::MatrixXd>    g = x;
  // auto y = a->function(m);
  // y = a->gradient(y);

  // std::cout << (*g.lock()).rows() << std::endl;
  // std::cout << y << std::endl;


/*
 #Eigen and shared_ptr
  std::shared_ptr<Eigen::MatrixXd> x = 
    std::make_shared<Eigen::MatrixXd>(4, 3);
  x->setZero();
  // concrete
  (*x)(0, 0) = 1;

  // random 
  Eigen::MatrixXd m = Eigen::MatrixXd::Random(4, 3);
  x = std::make_shared<Eigen::MatrixXd>(m);  // copy into shared_ptr

  // fill matrix
  (*x)
    <<
      0, 1, 0,
      0, 1, 1,
      1, 0, 0,
      0, 0, 1;
*/

  // Activations::create<Activations::Sigmoid>();


  // std::vector<std::unique_ptr<Layer>> MyLayers;
  // MyLayers.push_back(std::make_unique<SigmoidLayer>(3, 3, true));
  // MyLayers.push_back(std::make_unique<SigmoidLayer>(3, 2, true));
  // MyLayers.push_back(std::make_unique<SigmoidLayer>(2, 1, true));
  //
  // Eigen::MatrixXd x_input_train(4, 3);
  // x_input_train
  //   <<
  //     0, 1, 0,
  //     0, 1, 1,
  //     1, 0, 0,
  //     0, 0, 1;
  //
  // Eigen::MatrixXd y_targ_train(4, 1);
  // y_targ_train
  //   <<
  //     1,
  //     1,
  //     0,
  //     0;
  //
  // Eigen::MatrixXd x_input_test(2, 3);
  // x_input_test
  //   <<
  //     1, 1, 0,
  //     0, 0, 0;
  //
  // Eigen::MatrixXd y_targ_test(2, 1);
  // y_targ_test
  //   <<
  //     1,
  //     0;
  //
  //
  // std::cout << "\nTrain Input Data:\n" << x_input_train << std::endl;
  // std::cout << "\nTrain Target Data:\n" << y_targ_train << std::endl;
  //
  // std::cout << "\nTest Input Data:\n" << x_input_test << std::endl;
  // std::cout << "\nTest Target Data:\n" << y_targ_test << std::endl;
  //
  // std::cout << "_____________________________________" << std::endl;
  //
  // std::unique_ptr<Loss> myLoss = std::make_unique<MAE>();
  //
  // Model myModel(MyLayers, myLoss, 0.1);
  //
  // int epoch = 10;
  // int batch_size = 50;
  //
  // for (int i = 1; i <= epoch * batch_size; i++) {
  //   myModel.setInput(x_input_train.row(i % 4));
  //   myModel.setTarget(y_targ_train.row(i % 4));
  //   if (i % batch_size == 0) {
  //     myModel.forward();
  //     std::cout << "epoch - " << i / batch_size << "\tLoss:\t" << myModel.getLoss() << std::endl;
  //   }
  //   myModel.backward();
  // }
  //
  // for (size_t i = 0; i < 2; i++) {
  //   myModel.setInput(x_input_test.row(i));
  //   myModel.setTarget(y_targ_test.row(i));
  //   myModel.forward();
  //
  //   std::cout << "\n================================"     << std::endl;
  //   std::cout << "Input:\t\t"   << x_input_test.row(i)    << std::endl;
  //   std::cout << "Target:\t\t"  << y_targ_test.row(i)     << std::endl;
  //   std::cout << "Loss:\t\t"    << myModel.getLoss()      << std::endl;
  //   std::cout << "Predict:\t"   << myModel.getOutput()    << std::endl;
  // }

 
  return 0;
}
