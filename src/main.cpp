#include <vector>

#include <cmath>

#include "utils.hpp"
#include "nn.hpp"

Eigen::MatrixXd LossMAE (Eigen::MatrixXd target, Eigen::MatrixXd predict) {
  Eigen::MatrixXd difference = (target.array() - predict.array());
  return difference.array() / difference.size();
}

int main (int argc, char *argv[]) {

  std::vector<std::unique_ptr<Layer>> MyLayers;
  MyLayers.push_back(std::make_unique<SigmoidLayer>(3, 3, true));
  MyLayers.push_back(std::make_unique<SigmoidLayer>(3, 2, true));
  MyLayers.push_back(std::make_unique<SigmoidLayer>(2, 1, true));

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
      1, 1, 0,
      0, 0, 0;

  Eigen::MatrixXd y_targ_test(2, 1);
  y_targ_test
    <<
      1,
      0;


  std::cout << "\nTrain Input Data:\n" << x_input_train << std::endl;
  std::cout << "\nTrain Target Data:\n" << y_targ_train << std::endl;

  std::cout << "\nTest Input Data:\n" << x_input_test << std::endl;
  std::cout << "\nTest Target Data:\n" << y_targ_test << std::endl;

  std::cout << "_____________________________________" << std::endl;

  std::unique_ptr<Loss> myLoss = std::make_unique<MAE>();

  Model myModel(MyLayers, myLoss, 0.1);

  int epoch = 10;
  int batch_size = 50;

  for (int i = 1; i <= epoch * batch_size; i++) {
    myModel.setInput(x_input_train.row(i % 4));
    myModel.setTarget(y_targ_train.row(i % 4));
    if (i % batch_size == 0) {
      myModel.forward();
      std::cout << "epoch - " << i / batch_size << "\tLoss:\t" << myModel.getLoss() << std::endl;
    }
    myModel.backward();
  }

  for (size_t i = 0; i < 2; i++) {
    myModel.setInput(x_input_test.row(i));
    myModel.setTarget(y_targ_test.row(i));
    myModel.forward();

    std::cout << "\n================================"     << std::endl;
    std::cout << "Input:\t\t"   << x_input_test.row(i)    << std::endl;
    std::cout << "Target:\t\t"  << y_targ_test.row(i)     << std::endl;
    std::cout << "Loss:\t\t"    << myModel.getLoss()      << std::endl;
    std::cout << "Predict:\t"   << myModel.getOutput()    << std::endl;
  }

 
  return 0;
}
