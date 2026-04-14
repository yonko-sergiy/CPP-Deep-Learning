

#include "layer.h"

#include <boost/serialization/export.hpp>

BOOST_CLASS_EXPORT(Layers::Linear)

namespace Layers {

void Layer::setInput(Eigen::MatrixXd input) { this->input = input; }
void Layer::setDelta(Eigen::MatrixXd delta) { this->delta = delta; }

void Layer::bindNextLayer(std::shared_ptr<Layer> layer) { this->nextLayer = layer; }

//*std::shared_ptr<Layer>
// Layer::create() { return std::make_shared<Layer>(); };


Linear::Linear() {}
Linear::Linear(
  int input,
  int output,
  std::shared_ptr<Activations::Activation> activation,
  bool bias) : Layer() {
  this->weights = Eigen::MatrixXd::Random(input, output);
  this->activation = activation;
  if (bias) {
    this->biases = Eigen::MatrixXd::Random(1, output);
    this->bias = bias;
  }
}
//
//   this->input = Eigen::MatrixXd::Random(1, input);
//   this->linearOutput = Eigen::MatrixXd::Random(1, output);
//   // this->delta = 
// }
//
// void Linear::setInput(Eigen::MatrixXd input) { this->input = input; }
// void Linear::bindNextLayer(std::shared_ptr<Linear> layer) { this->nextLayer = layer; }
Eigen::MatrixXd Linear::getResult() {
  return activation->function(linearOutput);
}

std::shared_ptr<Linear> Linear::create(int input,
         int output, 
         std::shared_ptr<Activations::Activation> activation,
         bool bias) {
  return std::make_shared<Linear>(input, output, activation, bias);
}

Eigen::MatrixXd Linear::propagateDelta() { return (delta * weights.transpose()); }

void Linear::backward() {
  if (nextLayer) {
    delta = nextLayer->propagateDelta().array() * activation->gradient(linearOutput).array();
  }
}

void Linear::backward(Eigen::MatrixXd error) {
  this->delta = error.array() * activation->gradient(linearOutput).array();
}

void Linear::findBiases(double rate) {
  if (bias) { biases -= rate * delta.colwise().sum(); }
}

void Linear::findWeights(double rate) {
  weights -=  rate * (input.transpose() * delta);
}

//=======================
// void Linear::forward() {
//   Eigen::MatrixXd linearOuput = (*input.lock()) * weights;
//   if (bias) { linearOuput += biases.replicate((*input.lock()).rows(), 1); }
//   std::cout << linearOuput << std::endl;
//   this->output =
//     std::make_shared<Eigen::MatrixXd>(
//       activation->function(linearOuput)
//     );
// }
//=======================

void Linear::forward() {
  this->linearOutput = input * weights;
  if (bias) { this->linearOutput += biases.replicate(this->linearOutput.rows(), 1); }
  if (nextLayer) {
    nextLayer->setInput(
      activation->function(this->linearOutput)
    );
  }
}

void Linear::forward(Eigen::MatrixXd input) {
  setInput(input);
  linearOutput = this->input * weights;
  if (bias) { linearOutput += biases.replicate(linearOutput.rows(), 1); }
  if (nextLayer) {
    nextLayer->setInput(
      activation->function(linearOutput)
    );
  }
}

// =====================================
// template<class Archive>
// void Linear::serialize(Archive & ar, const unsigned int version) {
  // ar & input;
  // ar & linearOutput;
  // ar & weights;
  // ar & biases;
  // ar & delta;
  // ar & bias;
// }

// Explicit instantiation for supported archive types:
// template void Linear::serialize<boost::archive::text_oarchive>(boost::archive::text_oarchive&, unsigned int);
// template void Linear::serialize<boost::archive::text_iarchive>(boost::archive::text_iarchive&, unsigned int);
// =======================================


Sequential::Sequential() {};
Sequential::Sequential(const std::vector<std::shared_ptr<Layer>> & layers) {
  this->layers = layers;
  for (size_t i = 0; i < this->layers.size() - 1; i++) {
    this->layers[i]->bindNextLayer(this->layers[i + 1]);
  }
}

void Sequential::forward(Eigen::MatrixXd input) {
  layers[0]->setInput(input);
  for(auto layer : layers) {
    layer->forward();
  }
}

void Sequential::backward(Eigen::MatrixXd error) {
  bool first = true;
  for (int i = this->layers.size() - 1; i >= 0; i--) {
    if (first) {
      this->layers[i]->backward(error);
      first = false;
    } else {
      this->layers[i]->backward();
    }
  }
}

void Sequential::findBiases(double rate) {
  for(auto layer : layers) {
    layer->findBiases(rate);
  }
}

void Sequential::findWeights(double rate) {
  for(auto layer : layers) {
    layer->findWeights(rate);
  }
}

Eigen::MatrixXd Sequential::getResult() {
  return (*(layers.end() - 1))->getResult();
}

} // Layers end
