
// #include <boost/serialization/vector.hpp>
// #include <vector>
#include <fstream>

#include <cmath>

#include "activation.h"
#include "layer.h"
#include "loss.h"

#include <iostream>


// namespace boost {
// namespace serialization {
//
// // Serialize Eigen::MatrixXd
// template<class Archive>
// void serialize(Archive & ar, Eigen::MatrixXd & matrix, const unsigned int version) {
//     int rows = matrix.rows();
//     int cols = matrix.cols();
//
//     if (Archive::is_saving::value) {
//         ar & rows;
//         ar & cols;
//         // Save data as a vector
//         std::vector<double> data(matrix.data(), matrix.data() + rows * cols);
//         ar & data;
//     } else {
//         ar & rows;
//         ar & cols;
//         std::vector<double> data;
//         ar & data;
//         matrix.resize(rows, cols);
//         std::copy(data.begin(), data.end(), matrix.data());
//     }
// }
//
// } // namespace serialization
// } // namespace boost




// #include "nn.hpp"

// Eigen::MatrixXd LossMAE (Eigen::MatrixXd target, Eigen::MatrixXd predict) {
//   Eigen::MatrixXd difference = (target.array() - predict.array());
//   return difference.array() / difference.size();
// }

// Layer CreateLayer(SigmoidLayer, 3, 3, true) {
//   return shared_pointer
// };
// auto my_layer = CreateLayer(SigmoidLayer, 3, 3, true) {

int main (int argc, char *argv[]) {
  // auto a = Activations::Activation::create();
  // a = Activations::Sigmoid::create();
  // Eigen::MatrixXd m = Eigen::MatrixXd::Random(20, 1024);
  // Eigen::MatrixXd b = Eigen::MatrixXd::Random(20, 4);
  // Eigen::MatrixXd c = Eigen::MatrixXd::Random(8, 3);
  // std::shared_ptr<Eigen::MatrixXd> x =
  //   std::make_shared<Eigen::MatrixXd>(m);  // copy into shared_ptr
  //
  //

  auto sigmoid = Activations::Sigmoid::create();
  auto mae = Loss::MAE::create();
  
  auto model = Layers::Sequential({
    Layers::Linear::create(3, 3, sigmoid, true),
    Layers::Linear::create(3, 2, sigmoid, true),
    Layers::Linear::create(2, 1, sigmoid, true)
  });

  Eigen::MatrixXd in = Eigen::MatrixXd::Random(1, 3);
  Eigen::MatrixXd err = Eigen::MatrixXd::Random(1, 3);
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

 
  {
    std::ofstream ofs("model.dat");
    boost::archive::text_oarchive oa(ofs);
    oa << model;
  }
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
