#pragma once

#include <memory>
#include <Eigen/Dense>

#include "activation.h"
#include "serialization.h"


namespace Layers {

// Layer -- base class
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
  };
}; // end Layer -- base class


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

  Eigen::MatrixXd linearOutput;
}; // end Linear


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
}; // end Sequential

} // namespace Layers
