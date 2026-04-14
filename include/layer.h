#pragma once

#include <memory>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/make_shared.hpp>
#include <boost/serialization/nvp.hpp>

#include <boost/serialization/vector.hpp>
#include <vector>
#include <fstream>

#include "activation.h"
#include "serialization.h"

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


namespace Layers {


class Layer {
public:
  Layer() = default;
  virtual ~Layer() = default;

  virtual void forward() = 0;
  virtual void forward(Eigen::MatrixXd input) = 0;
  virtual Eigen::MatrixXd getResult() = 0;

  virtual void backward() = 0;
  virtual void backward(Eigen::MatrixXd error) = 0;
  virtual Eigen::MatrixXd propagateDelta() = 0;


  virtual void findBiases(double rate) = 0 ;
  virtual void findWeights(double rate) = 0;



  void bindNextLayer(std::shared_ptr<Layer> layer);

  void setInput(Eigen::MatrixXd input);
  void setDelta(Eigen::MatrixXd delta);

  //* static std::shared_ptr<Layer> create();
  // Layer(int input, int output, bool bias);

  // virtual Eigen::MatrixXd activation() = 0;
  // virtual Eigen::MatrixXd gradient() = 0;


  // void setInput   (Eigen::MatrixXd input);
  // void setWeights (Eigen::MatrixXd delta);
  // void setBiases  (Eigen::MatrixXd biases);
  //
  // void setDelta   (Eigen::MatrixXd delta);
  // void setGrads   (Eigen::MatrixXd grads);
  //
  //
  // Eigen::MatrixXd getInput   ();
  // Eigen::MatrixXd getWeights ();
  // Eigen::MatrixXd getBiases  ();
  //
  // Eigen::MatrixXd getDelta   ();
  // Eigen::MatrixXd getGrads   ();
  //
  // bool isBiases();

protected:
  std::shared_ptr<Activations::Activation> activation;
  std::shared_ptr<Layer> nextLayer;

  Eigen::MatrixXd input;

  Eigen::MatrixXd weights;
  Eigen::MatrixXd biases;

  Eigen::MatrixXd delta;

  bool bias = false;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
      ar & weights;
      ar & biases;
      ar & bias;
      ar & activation;
      ar & nextLayer;
      // ar & BOOST_SERIALIZATION_NVP(bias);

      // ar & BOOST_SERIALIZATION_NVP(weights);
      // ar & BOOST_SERIALIZATION_NVP(biases);
      // ar & BOOST_SERIALIZATION_NVP(bias);
      // ar & BOOST_SERIALIZATION_NVP(activation);
      // ar & BOOST_SERIALIZATION_NVP(nextLayer);
  };

};



class Linear : public Layer {
public:
  Linear();
  Linear(
    int input,
    int output,
    std::shared_ptr<Activations::Activation> activation,
    bool bias);

  void forward() override;
  void forward(Eigen::MatrixXd input) override;
  Eigen::MatrixXd getResult() override;

  void backward() override;
  void backward(Eigen::MatrixXd error) override;
  Eigen::MatrixXd propagateDelta() override;

  void findBiases(double rate) override;
  void findWeights(double rate) override;

//   void bindNextLayer(std::shared_ptr<Linear> layer);
//   void setInput(Eigen::MatrixXd input);
//   void setDelta(Eigen::MatrixXd delta);

  static std::shared_ptr<Linear> create(int input,
        int output, 
        std::shared_ptr<Activations::Activation> activation,
        bool bias);

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
      ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
  };
//   friend class boost::serialization::access;
//   template<class Archive>
//   void serialize(Archive & ar, const unsigned int version) {
//       ar & weights;
//       ar & biases;
//       ar & bias;
//       ar & activation;
//       ar & nextLayer;
//       // ar & BOOST_SERIALIZATION_NVP(bias);
//   };
//
//
//   std::shared_ptr<Activations::Activation> activation;
//   std::shared_ptr<Linear> nextLayer;
//
//   Eigen::MatrixXd input;
  Eigen::MatrixXd linearOutput;
//
//   Eigen::MatrixXd weights;
//   Eigen::MatrixXd biases;
//
//   Eigen::MatrixXd delta;
//
//   bool bias = false;
};


class Sequential {
public:
  Sequential();
  Sequential(const std::vector<std::shared_ptr<Layer>> & layers);

  void forward(Eigen::MatrixXd input);
  void backward(Eigen::MatrixXd error);

  void findBiases(double rate = 0.1);
  void findWeights(double rate = 0.1);

  Eigen::MatrixXd getResult();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & layers;
  };
  std::vector<std::shared_ptr<Layer>> layers;
};

}

