
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


// Identity
class Identity : public Activation {
public:
  Eigen::MatrixXd function(Eigen::MatrixXd X) const override;
  Eigen::MatrixXd gradient(Eigen::MatrixXd X) const override;

  static std::shared_ptr<Identity> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Activation);
  };
}; // end Identity


// Tanh
class Tanh : public Activation {
public:
  Eigen::MatrixXd function(Eigen::MatrixXd X) const override;
  Eigen::MatrixXd gradient(Eigen::MatrixXd X) const override;

  static std::shared_ptr<Tanh> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Activation);
  };
}; // end Tanh


// Sigmoid
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


// Softsign
class Softsign : public Activation {
public:
  Eigen::MatrixXd function(Eigen::MatrixXd X) const override;
  Eigen::MatrixXd gradient(Eigen::MatrixXd X) const override;

  static std::shared_ptr<Softsign> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Activation);
  };
}; // end Softsign


// Softplus
class Softplus : public Activation {
public:
  Eigen::MatrixXd function(Eigen::MatrixXd X) const override;
  Eigen::MatrixXd gradient(Eigen::MatrixXd X) const override;

  static std::shared_ptr<Softplus> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Activation);
  };
}; // end Softplus


// SiLU
class SiLU : public Activation {
public:
  Eigen::MatrixXd function(Eigen::MatrixXd X) const override;
  Eigen::MatrixXd gradient(Eigen::MatrixXd X) const override;

  static std::shared_ptr<SiLU> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Activation);
  };
}; // end SiLU


// Gaussian
class Gaussian : public Activation {
public:
  Eigen::MatrixXd function(Eigen::MatrixXd X) const override;
  Eigen::MatrixXd gradient(Eigen::MatrixXd X) const override;

  static std::shared_ptr<Gaussian> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Activation);
  };
}; // end Gaussian


// Sinusoid
class Sinusoid : public Activation {
public:
  Eigen::MatrixXd function(Eigen::MatrixXd X) const override;
  Eigen::MatrixXd gradient(Eigen::MatrixXd X) const override;

  static std::shared_ptr<Sinusoid> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Activation);
  };
}; // end Sinusoid

} // namespace Activations
} // namespace NN
