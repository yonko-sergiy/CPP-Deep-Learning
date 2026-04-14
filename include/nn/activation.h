
#pragma once

#include <memory>
#include <Eigen/Dense>

#include "serialization.h"


namespace NN {
namespace Activations {

// Activation -- base class
class Activation {
public:
  virtual Eigen::MatrixXd function(Eigen::MatrixXd X) const;
  virtual Eigen::MatrixXd gradient(Eigen::MatrixXd X) const;
  static std::shared_ptr<Activation> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {};
}; // end Activation -- base class


class Sigmoid : public Activation {
public:

  Eigen::MatrixXd function(Eigen::MatrixXd X) const override;
  Eigen::MatrixXd gradient(Eigen::MatrixXd X) const override;

  static std::shared_ptr<Sigmoid> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Activation);
  };
}; // end Sigmoid

} // namespace Activations
} // namespace NN
