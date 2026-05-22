
#include <boost/serialization/export.hpp>

#include "nn/activation.h"


BOOST_CLASS_EXPORT(NN::Activations::Identity)
BOOST_CLASS_EXPORT(NN::Activations::Tanh)
BOOST_CLASS_EXPORT(NN::Activations::Sigmoid)
BOOST_CLASS_EXPORT(NN::Activations::Softsign)
BOOST_CLASS_EXPORT(NN::Activations::Softplus)
BOOST_CLASS_EXPORT(NN::Activations::SiLU)
BOOST_CLASS_EXPORT(NN::Activations::Gaussian)
BOOST_CLASS_EXPORT(NN::Activations::Sinusoid)

namespace NN {
namespace Activations {

// Activation -- base class
std::shared_ptr<Activation> 
Activation::create() { return std::make_shared<Activation>(); };
Eigen::MatrixXd
Activation::function(Eigen::MatrixXd X) const { return X; };
Eigen::MatrixXd
Activation::gradient(Eigen::MatrixXd X) const { return X; };
// end Activation -- base class


// Identity
std::shared_ptr<Identity>
Identity::create() { return std::make_shared<Identity>(); };

Eigen::MatrixXd
Identity::function(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return x;
  });
  return tmp_output;
};

Eigen::MatrixXd
Identity::gradient(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return x/x; // 1
  });
  return tmp_output;
};
// end Identity


// Tanh
std::shared_ptr<Tanh>
Tanh::create() { return std::make_shared<Tanh>(); };

Eigen::MatrixXd
Tanh::function(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return (std::exp(x) - std::exp(-x)) /
           (std::exp(x) + std::exp(-x));
  });
  return tmp_output;
};

Eigen::MatrixXd
Tanh::gradient(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = function(X).unaryExpr([](double x) {
    return x * (1 - x); // 1
  });
  return tmp_output;
};
// end Tanh


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


// Softsign
std::shared_ptr<Softsign>
Softsign::create() { return std::make_shared<Softsign>(); };

Eigen::MatrixXd
Softsign::function(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return x / (1.0 + std::abs(x));
  });
  return tmp_output;
};

Eigen::MatrixXd
Softsign::gradient(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return 1.0 / std::pow(1.0 + std::abs(x), 2);
  });
  return tmp_output;
};
// end Softsign


// Softplus
std::shared_ptr<Softplus>
Softplus::create() { return std::make_shared<Softplus>(); };

Eigen::MatrixXd
Softplus::function(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return std::log(1.0 + std::exp(x));
  });
  return tmp_output;
};

Eigen::MatrixXd
Softplus::gradient(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return 1.0 / (1.0 + std::exp(-x));
  });
  return tmp_output;
};
// end Softplus


// SiLU
std::shared_ptr<SiLU>
SiLU::create() { return std::make_shared<SiLU>(); };

Eigen::MatrixXd
SiLU::function(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return x / (1.0 + std::exp(-x));
  });
  return tmp_output;
};

Eigen::MatrixXd
SiLU::gradient(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return (1 + std::exp(-x) + x * std::exp(-x)) /
           pow(1 + std::exp(-x), 2);
  });
  return tmp_output;
};
// end SiLU


// Gaussian
std::shared_ptr<Gaussian>
Gaussian::create() { return std::make_shared<Gaussian>(); };

Eigen::MatrixXd
Gaussian::function(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return std::exp(-std::pow(x, 2));
  });
  return tmp_output;
};

Eigen::MatrixXd
Gaussian::gradient(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return -2 * x * std::exp(-std::pow(x, 2));
  });
  return tmp_output;
};
// end Gaussian


// Sinusoid
std::shared_ptr<Sinusoid>
Sinusoid::create() { return std::make_shared<Sinusoid>(); };

Eigen::MatrixXd
Sinusoid::function(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return std::sin(x);
  });
  return tmp_output;
};

Eigen::MatrixXd
Sinusoid::gradient(Eigen::MatrixXd X) const {
  Eigen::MatrixXd tmp_output = X.unaryExpr([](double x) {
    return std::cos(x);
  });
  return tmp_output;
};
// end Sinusoid

} // namespace Activations
} // namespace NN
