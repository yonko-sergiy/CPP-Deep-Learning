
#include "nn.hpp"

Eigen::MatrixXd SigmoidLayer::activation() {
  Eigen::MatrixXd tmp_output = this->input * this->weights;
  if (Bias) { tmp_output += biases.replicate(this->input.rows(), 1); }
  tmp_output = tmp_output.unaryExpr([](float x) {
    return 1.0 / (1.0 + std::exp(-x));
  });
  return tmp_output;
}

Eigen::MatrixXd SigmoidLayer::gradient() {
  Eigen::MatrixXd gradient_out;
  gradient_out = this->input * this->weights;
  if (Bias) { gradient_out += biases.replicate(this->input.rows(), 1); }
  gradient_out = gradient_out.unaryExpr([](float x) {
      return (1.0 / (1.0 + std::exp(-x))) * (1 - (1.0 / (1.0 + std::exp(-x))));
  }).array();
  return gradient_out;
}

Eigen::MatrixXd SoftplusLayer::activation() {
  Eigen::MatrixXd tmp_output = this->input * this->weights;
  if (Bias) { tmp_output += biases.replicate(this->input.rows(), 1); }
  tmp_output = tmp_output.unaryExpr([](float x) {
    return std::log(1.0 + std::exp(x));
  });
  return tmp_output;
}

Eigen::MatrixXd SoftplusLayer::gradient() {
  Eigen::MatrixXd gradient_out;
  gradient_out = this->input * this->weights;
  if (Bias) { gradient_out += biases.replicate(this->input.rows(), 1); }
  gradient_out = gradient_out.unaryExpr([](float x) {
      return 1.0 / (1.0 + std::exp(-x));
  }).array();
  return gradient_out;
}

void Layer::setInput  (Eigen::MatrixXd input)   { this->input   = input;   }
void Layer::setWeights(Eigen::MatrixXd weights) { this->weights = weights; }
void Layer::setBiases (Eigen::MatrixXd biases)  { this->biases  = biases;  }
void Layer::setDelta  (Eigen::MatrixXd delta)   { this->delta   = delta;   }
void Layer::setGrads  (Eigen::MatrixXd grads)   { this->grads   = grads;   }

Eigen::MatrixXd Layer::getInput   () { return this->input;   }
Eigen::MatrixXd Layer::getWeights () { return this->weights; }
Eigen::MatrixXd Layer::getBiases  () { return this->biases;  }
Eigen::MatrixXd Layer::getDelta   () { return this->delta;   }
Eigen::MatrixXd Layer::getGrads   () { return this->grads;   }

bool Layer::isBiases() { return this->Bias; };


Layer::Layer(int input, int output, bool bias = false) {
  this->weights = Eigen::MatrixXd::Random(input, output);
  if (bias) {
    this->biases = Eigen::MatrixXd::Random(1, output);
    this->Bias = bias;
  }
};

Loss::Loss() {};

// Eigen::MatrixXd function (Eigen::MatrixXd target, Eigen::MatrixXd predict);
Eigen::MatrixXd MAE::function (Eigen::MatrixXd target, Eigen::MatrixXd predict) {
  Eigen::MatrixXd difference = (target.array() - predict.array()).cwiseAbs();
  return (difference.array() / difference.size());
};

Eigen::MatrixXd MSE::function (Eigen::MatrixXd target, Eigen::MatrixXd predict) {
  Eigen::MatrixXd difference = (target.array() - predict.array()).cwiseSquare();
  return difference.array() / difference.size();
};

Model::Model(std::vector<std::unique_ptr<Layer>>& layers, double rate = 0.1) { this->Layers = std::move(layers); this->rate = rate; };

void Model::setInput(Eigen::MatrixXd X) { this->Input = X; }
void Model::setTarget(Eigen::MatrixXd Y) {
   // this->Layers.back()->setDelta( (Y.array() - this->Output.array()) * this->Layers.back()->gradient().array() ); // Formula 1
  // this->Layers.back()->setDelta((Y.array() - this->Output.array())); // Formula 2
  this->Target = Y;
}

Model::Model(std::vector<std::unique_ptr<Layer>>& layers, std::unique_ptr<Loss>& loss, double rate = 0.1) {
  this->Layers = std::move(layers); this->rate = rate;
  this->loss = std::move(loss);
};

void Model::forward() {
  Eigen::MatrixXd tmp_input = this->Input;
  for (size_t i = 0; i < this->Layers.size(); i++) {
    this->Layers[i]->setInput(tmp_input);
    tmp_input = this->Layers[i]->activation();
  }
  Output = tmp_input;
}

void Model::backward() {

  for (size_t i = 0; i < 10; i++) {
    this->forward();
    for (int i = this->Layers.size() - 1; i >= 0; i--) { this->_FindDelta (i); }
    for (int i = this->Layers.size() - 1; i >= 0; i--) { this->_FindGrad  (i); }
    for (int i = this->Layers.size() - 1; i >= 0; i--) { this->_FindWeight(i); }
  }

  this->forward();
  for (int i = this->Layers.size() - 1; i >= 0; i--) { this->_FindDelta (i); }
  for (int i = this->Layers.size() - 1; i >= 0; i--) { this->_FindBias  (i); }
}

void Model::_FindDelta(int layer_id) {
  if ( layer_id == (this->Layers.size() - 1) ) {
    // this->Layers.back()->setDelta( (this->Target.array() - this->Output.array()) * this->Layers.back()->gradient().array() ); // Formula 1
    this->Layers.back()->setDelta((this->Target.array() - this->Output.array())); // Formula 2
  } else if (layer_id < 0 || layer_id > this->Layers.size() - 1) { std::cout << "Layer out range" << std::endl; }
  else {
    Eigen::MatrixXd delta = this->Layers[layer_id+1]->getDelta();
    Eigen::MatrixXd weights = this->Layers[layer_id+1]->getWeights();
    Eigen::MatrixXd gradient = this->Layers[layer_id]->gradient();
    // this->Layers[layer_id]->setDelta( (delta * weights.transpose()).array() * gradient.array() ); // Formula 1
    this->Layers[layer_id]->setDelta( delta * weights.transpose() ); // Formula 2
  }
}

void Model::_FindGrad(int layer_id) {
  Eigen::MatrixXd delta = this->Layers[layer_id]->getDelta();
  Eigen::MatrixXd input = this->Layers[layer_id]->getInput();
  Eigen::MatrixXd gradient = this->Layers[layer_id]->gradient();

  int weights_rows = this->Layers[layer_id]->getWeights().rows();

  // this->Layers[layer_id]->setGrads( input.transpose() * delta ); // Formula 1
  this->Layers[layer_id]->setGrads( (input.transpose() * delta).array() * gradient.replicate(weights_rows, 1).array() ); // Formula 2
  // Eigen::MatrixXd grads_tmp = gradient * input;
}

void Model::_FindWeight(int layer_id) {
  Eigen::MatrixXd grads = this->Layers[layer_id]->getGrads();
  Eigen::MatrixXd weights = this->Layers[layer_id]->getWeights();

  int weights_rows = this->Layers[layer_id]->getWeights().rows();

  // this->Layers[layer_id]->setWeights( weights.array() - (rate * grads).array() * weights.array() ); // Formula 1

  this->Layers[layer_id]->setWeights( weights + rate * grads ); // Formula 2
}

void Model::_FindBias(int layer_id) {
  if (this->Layers[layer_id]->isBiases()) {
    Eigen::MatrixXd delta = this->Layers[layer_id]->getDelta();
    Eigen::MatrixXd biases = this->Layers[layer_id]->getBiases();
    // std::cout << biases.cols() << std::endl;
    // std::cout << delta.cols() << std::endl;
    this->Layers[layer_id]->setBiases( biases.array() + this->rate * delta.colwise().sum().array() );
  }
}

Eigen::MatrixXd Model::getOutput() { return this->Output; }
Eigen::MatrixXd Model::getLoss()   {
  return loss->function(this->Target, this->Output);
};


