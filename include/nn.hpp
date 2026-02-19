
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Dense>


class Layer {
public:
  Layer(int input, int output, bool bias);

  virtual Eigen::MatrixXd activation() = 0;
  virtual Eigen::MatrixXd gradient() = 0;
  
  void setInput   (Eigen::MatrixXd input);
  void setWeights (Eigen::MatrixXd delta);
  void setBiases  (Eigen::MatrixXd biases);

  void setDelta   (Eigen::MatrixXd delta);
  void setGrads   (Eigen::MatrixXd grads);


  Eigen::MatrixXd getInput   ();
  Eigen::MatrixXd getWeights ();
  Eigen::MatrixXd getBiases  ();

  Eigen::MatrixXd getDelta   ();
  Eigen::MatrixXd getGrads   ();

  bool isBiases();


protected:
  Eigen::MatrixXd input;
  Eigen::MatrixXd weights;
  Eigen::MatrixXd biases;

  Eigen::MatrixXd delta;
  Eigen::MatrixXd grads;

  bool Bias = false;
};


class SigmoidLayer : public Layer {
public:
  using Layer::Layer;
  Eigen::MatrixXd activation() override;
  Eigen::MatrixXd gradient() override;
};


class SoftplusLayer : public Layer{
public:
  using Layer::Layer;
  Eigen::MatrixXd activation() override;
  Eigen::MatrixXd gradient() override;
};

class Loss {
public:
  Loss();
  virtual Eigen::MatrixXd function (Eigen::MatrixXd target, Eigen::MatrixXd predict) = 0;
};

class MAE : public Loss {
public:
  using Loss::Loss;
  Eigen::MatrixXd function (Eigen::MatrixXd target, Eigen::MatrixXd predict) override;
};

class MSE : public Loss {
public:
  using Loss::Loss;
  Eigen::MatrixXd function (Eigen::MatrixXd target, Eigen::MatrixXd predict) override;
};

class Model {
  public:
    Model(std::vector<std::unique_ptr<Layer>>& layers, double rate);
    Model(std::vector<std::unique_ptr<Layer>>& layers, std::unique_ptr<Loss>& loss, double rate);
    void setInput(Eigen::MatrixXd X);
    void setTarget(Eigen::MatrixXd Y);

    void forward();
    void backward();

    void _FindDelta   ( int layer_id );
    void _FindGrad    ( int layer_id );
    void _FindWeight  ( int layer_id );
    void _FindBias    ( int layer_id );

    Eigen::MatrixXd getOutput();
    Eigen::MatrixXd getLoss();


  private:
    std::vector<std::unique_ptr<Layer>> Layers;
    std::unique_ptr<Loss> loss;

    Eigen::MatrixXd Input;
    Eigen::MatrixXd Output;
    Eigen::MatrixXd Target;

    double rate = 1;
};

