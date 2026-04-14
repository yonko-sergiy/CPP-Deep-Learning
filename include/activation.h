
#pragma once

#include <memory>
#include <Eigen/Dense>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/make_shared.hpp>
#include <boost/serialization/nvp.hpp>

#include <iostream>

namespace Activations {

// #Activation
class Activation {
public:
  virtual Eigen::MatrixXd function(Eigen::MatrixXd X) const;
  virtual Eigen::MatrixXd gradient(Eigen::MatrixXd X) const;
  static std::shared_ptr<Activation> create();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {};
}; // #Activation end


// #Sigmoid
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

}; // #Sigmoid end


} // end Activations

