
#include <boost/serialization/export.hpp>

#include "activation.h"


BOOST_CLASS_EXPORT(Activations::Sigmoid)

namespace Activations {

// Activation -- base class
std::shared_ptr<Activation> 
Activation::create() { return std::make_shared<Activation>(); };
Eigen::MatrixXd
Activation::function(Eigen::MatrixXd X) const { return X; };
Eigen::MatrixXd
Activation::gradient(Eigen::MatrixXd X) const { return X; };
// end Activation -- base class


// Sigmoid
std::shared_ptr<Sigmoid>
Sigmoid::create() { return std::make_shared<Sigmoid>(); };

Eigen::MatrixXd
Sigmoid::function(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return 1.0 / (1.0 + std::exp(-x));
  });
  return tmp_output;
};

Eigen::MatrixXd
Sigmoid::gradient(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = function(X).unaryExpr([](double x) {
    return (x * (1 - x));
  });
  return tmp_output;
};
// end Sigmoid

}; // namespace Activations
