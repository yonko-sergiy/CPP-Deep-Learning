

#include <Eigen/Dense>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "nn/activation.h"
#include "nn/layer.h"
#include "nn/loss.h"


using namespace NN;


class TitanicDataset {
public:
  TitanicDataset(std::string filename);
  ~TitanicDataset();

  Eigen::MatrixXd getBatchOfTargetsTrain(int batch_size);
  Eigen::MatrixXd getBatchOfTargetsTests(int batch_size);

  Eigen::MatrixXd getBatchOfDatasetTrain(int batch_size);
  Eigen::MatrixXd getBatchOfDatasetTests(int batch_size);

  int getSizeOfTrain();
  int getSizeOfTests();

  void updateSeed();

private:
  Eigen::MatrixXd targets_train = Eigen::MatrixXd::Zero(712, 1);
  Eigen::MatrixXd dataset_train = Eigen::MatrixXd::Zero(712, 9);

  Eigen::MatrixXd targets_tests = Eigen::MatrixXd::Zero(179, 1);
  Eigen::MatrixXd dataset_tests = Eigen::MatrixXd::Zero(179, 9);

  int seed = 0;
};


void TitanicDataset::updateSeed() { seed = std::time(0); }
int TitanicDataset::getSizeOfTrain() { return targets_train.rows(); }
int TitanicDataset::getSizeOfTests() { return targets_tests.rows(); }

Eigen::MatrixXd TitanicDataset::getBatchOfTargetsTrain(int batch_size = 1) {
  Eigen::MatrixXd batch(batch_size, targets_train.cols());
  std::srand(seed);
  for (size_t i = 0; i < batch_size; i++) {
    int r = std::rand() % targets_train.rows();
    batch.block(i, 0, 1, targets_train.cols()) = targets_train.block(r, 0, 1, targets_train.cols());
  }
  return batch;
}

Eigen::MatrixXd TitanicDataset::getBatchOfTargetsTests(int batch_size = 1) {
  Eigen::MatrixXd batch(batch_size, targets_tests.cols());
  std::srand(seed);
  for (size_t i = 0; i < batch_size; i++) {
    int r = std::rand() % targets_tests.rows();
    batch.block(i, 0, 1, targets_tests.cols()) = targets_tests.block(r, 0, 1, targets_tests.cols());
  }
  return batch;
}

Eigen::MatrixXd TitanicDataset::getBatchOfDatasetTrain(int batch_size = 1) {
  Eigen::MatrixXd batch(batch_size, dataset_train.cols());
  std::srand(seed);
  for (size_t i = 0; i < batch_size; i++) {
    int r = std::rand() % dataset_train.rows();
    batch.block(i, 0, 1, dataset_train.cols()) = dataset_train.block(r, 0, 1, dataset_train.cols());
  }
  return batch;
}

Eigen::MatrixXd TitanicDataset::getBatchOfDatasetTests(int batch_size = 1) {
  Eigen::MatrixXd batch(batch_size, dataset_tests.cols());
  std::srand(seed);
  for (size_t i = 0; i < batch_size; i++) {
    int r = std::rand() % dataset_tests.rows();
    batch.block(i, 0, 1, dataset_tests.cols()) = dataset_tests.block(r, 0, 1, dataset_tests.cols());
  }
  return batch;
}



TitanicDataset::TitanicDataset(std::string filename = "") {
  std::vector<std::string> paths = {
      filename,
      "./datasets/titanic.csv",
      "../datasets/titanic.csv",
      "../../datasets/titanic.csv",
      "./titanic.csv",
      "../titanic.csv",
      "../../titanic.csv"
  };

  std::ifstream file;

  std::cout << "Trying to open dataset..." << std::endl;
  for(auto filename : paths) {
    file = std::ifstream(filename);
    if (file.is_open()) {
      std::cout << "File " << filename << " has opened." << std::endl;
      break;
    } else {
      std::cerr << "Error opening dataset: " << filename << std::endl;
    }
  }


  std::string line;
  std::vector<int> surived;
  std::vector<int> Pclass;
  std::vector<int> gender;
  std::vector<int> age;
  std::vector<int> SibSp;
  std::vector<int> Parch;
  std::vector<float> Fare;
  std::vector<int> EmbarkedS;
  std::vector<int> EmbarkedQ;
  std::vector<int> EmbarkedC;
  int i = 0;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> row;

    while (std::getline(ss, cell, ',')) {
        row.push_back(cell);
    }
    if (i == 0) {
      i += 1;
      continue;
    }

    surived.push_back(std::stod(row[1]));
    Pclass.push_back(std::stod(row[2]));
    if (row[5] == "male") gender.push_back(1);
    else gender.push_back(0);
    if (row[6].empty()) age.push_back(0);
    else age.push_back(std::stod(row[6]));
    SibSp.push_back(std::stod(row[7]));
    Parch.push_back(std::stod(row[8]));
    Fare.push_back(std::stod(row[10]));

    if (row.size() == 13) {
      if (row[12] == "Q") EmbarkedQ.push_back(1);
      else EmbarkedQ.push_back(0);
      if (row[12] == "S") EmbarkedS.push_back(1);
      else EmbarkedS.push_back(0);
      if (row[12] == "C") EmbarkedC.push_back(1);
      else EmbarkedC.push_back(0);
    } else {
      EmbarkedQ.push_back(0);
      EmbarkedS.push_back(0);
      EmbarkedC.push_back(0);
    }
  }
  file.close();

  for(int i = 0; i < surived.size(); i++) {
    if (i % 5 != 0) {
      targets_train(i-1-i/5, 0) = surived[i];

      dataset_train(i-1-i/5, 0) = (double)Pclass[i] / 3.0;
      dataset_train(i-1-i/5, 1) = (double)gender[i];
      dataset_train(i-1-i/5, 2) = (double)age[i]    / 100;
      dataset_train(i-1-i/5, 3) = (double)SibSp[i]  / 10;
      dataset_train(i-1-i/5, 4) = (double)Parch[i]  / 10;
      dataset_train(i-1-i/5, 5) = (double)Fare[i]   / 512;
      dataset_train(i-1-i/5, 6) = (double)EmbarkedQ[i];
      dataset_train(i-1-i/5, 7) = (double)EmbarkedS[i];
      dataset_train(i-1-i/5, 8) = (double)EmbarkedC[i];
    } else {
      targets_tests(i/5, 0) = surived[i];

      dataset_tests(i/5, 0) = (double)Pclass[i] / 3.0;
      dataset_tests(i/5, 1) = (double)gender[i];
      dataset_tests(i/5, 2) = (double)age[i]    / 100;
      dataset_tests(i/5, 3) = (double)SibSp[i]  / 10;
      dataset_tests(i/5, 4) = (double)Parch[i]  / 10;
      dataset_tests(i/5, 5) = (double)Fare[i]   / 512;
      dataset_tests(i/5, 6) = (double)EmbarkedQ[i];
      dataset_tests(i/5, 7) = (double)EmbarkedS[i];
      dataset_tests(i/5, 8) = (double)EmbarkedC[i];
    }
  }


};

TitanicDataset::~TitanicDataset() {}


class TitanicModel {
public:
  TitanicModel(std::string modelPath, std::string datasetPath);
  TitanicModel(TitanicModel &&) = default;
  TitanicModel(const TitanicModel &) = default;
  TitanicModel &operator=(TitanicModel &&) = default;
  TitanicModel &operator=(const TitanicModel &) = default;
  ~TitanicModel();

  void train(int batchSize, int checkAtEpoch, int epoch, double rate);
  void tests(int batchSize);
  void save(std::string filename);
  Eigen::MatrixXd predict(Eigen::MatrixXd input);

private:
  Layers::Sequential model;
  Loss::MAE mae;
  TitanicDataset dataset;
};

TitanicModel::TitanicModel(std::string modelPath = "", std::string datasetPath = "") {
  mae = Loss::MAE();
  std::vector<std::string> paths = {
    modelPath,
    "./models/titanic.dat",
    "../models/titanic.dat",
    "../../models/titanic.dat",
    "./titanic.dat",
    "../titanic.dat",
    "../../titanic.dat"
  };

  {
    std::cout << "Trying to open model..." << std::endl;
    std::ifstream ifs;
    for(auto filename : paths) {
      ifs = std::ifstream(filename);
      if (ifs.is_open()) {
        boost::archive::text_iarchive ia(ifs);
        ia >> model;
        std::cout << "Model " << filename << " has opened." << std::endl;
        break;
      } else {
        std::cerr << "Error opening model: " << filename << std::endl;
      }
    }
    if (!ifs.is_open()) {
      std::cout << "Creating new model..." << std::endl;
      auto sigmoid = Activations::Sigmoid::create();
      model = Layers::Sequential({
        Layers::Linear::create(9, 18, sigmoid, true),
        Layers::Linear::create(18, 9, sigmoid, true),
        Layers::Linear::create(9, 9, sigmoid, true),
        Layers::Linear::create(9, 1, sigmoid, true)
      });
      std::cout << "Model is created!" << std::endl;
    }
  }

  if (!datasetPath.empty()) {
    dataset = TitanicDataset(datasetPath);
  }
}

void TitanicModel::train(int batchSize = 64, int checkAtEpoch = 100, int epoch = 1000, double rate = 0.001) {

  Eigen::MatrixXd batchTargetsTrain;
  Eigen::MatrixXd batchTargetsTests;

  Eigen::MatrixXd batchDatasetTrain;
  Eigen::MatrixXd batchDatasetTests;

  int datasetSize   = dataset.getSizeOfTrain();
  int iterPerEpoch  = datasetSize / batchSize;
  double loss       = 1;

  std::cout << "\n===========DATA===========" << std::endl;
  std::cout << "Dataset_train_size:--:" << datasetSize              << std::endl;
  std::cout << "Dataset_tests_size:--:" << dataset.getSizeOfTests() << std::endl;
  std::cout << "Batch_size:----------:" << batchSize                << std::endl;
  std::cout << "Check_at_epoch:------:" << checkAtEpoch             << std::endl;
  std::cout << "Epoch:---------------:" << epoch                    << std::endl;
  std::cout << "Learning_rate:-------:" << rate                     << std::endl;

  std::cout << "\n#Training..." << std::endl;

  for (size_t i = 0; i <= epoch; i++) {
    dataset.updateSeed();
    // batchDatasetTrain = dataset.getBatchOfDatasetTrain(batchSize);
    // batchTargetsTrain = dataset.getBatchOfTargetsTrain(batchSize);
    batchDatasetTests = dataset.getBatchOfDatasetTests(batchSize);
    batchTargetsTests = dataset.getBatchOfTargetsTests(batchSize);

    for (size_t j = 0; j <= iterPerEpoch; j++) {
      dataset.updateSeed();
      batchDatasetTrain = dataset.getBatchOfDatasetTrain(batchSize);
      batchTargetsTrain = dataset.getBatchOfTargetsTrain(batchSize);
      model.forward(batchDatasetTrain);
      model.backward(model.getResult() - batchTargetsTrain);
      model.findWeights(rate);
    }

    for (size_t j = 0; j <= iterPerEpoch; j++) {
      dataset.updateSeed();
      batchDatasetTrain = dataset.getBatchOfDatasetTrain(batchSize);
      batchTargetsTrain = dataset.getBatchOfTargetsTrain(batchSize);
      model.forward(batchDatasetTrain);
      model.backward(model.getResult() - batchTargetsTrain);
      model.findBiases(rate);
    }

    if (checkAtEpoch != 0 && i % checkAtEpoch == 0) {
      model.forward(batchDatasetTests);
      loss = mae.function(model.getResult(), batchTargetsTests);
      std::cout << "Epoch: " << i << "/" << epoch << "\t\tLoss: " << loss << std::endl;
    }
  }
  std::cout << "\nTraining is complete!" << std::endl;
}

void TitanicModel::tests(int batchSize = 16) {
  dataset.updateSeed();
  Eigen::MatrixXd batchDatasetTests = dataset.getBatchOfDatasetTests(batchSize);
  Eigen::MatrixXd batchTargetsTests = dataset.getBatchOfTargetsTests(batchSize);

  model.forward(batchDatasetTests);
  Eigen::MatrixXd result = model.getResult();

  int rows = batchTargetsTests.rows();
  int cols = batchDatasetTests.cols() +
             batchTargetsTests.cols() +
             result.cols();
  Eigen::MatrixXd table(rows, cols);
  table << batchDatasetTests, batchTargetsTests, result;


  std::string table_cols = "|";
  std::string table_headers = "|Pclass   |gender   |age      |SibSp    |Parch    |Fare     |EmbarkedQ|EmbarkedQ|EmbarkedQ|Survied  |Chance   |";
  for (size_t i = 0; i < 11; i++) {
    table_cols += "---------|";
  }
  std::cout << table_cols    << std::endl;
  std::cout << table_headers << std::endl;
  std::cout << table_cols    << std::endl;
  std::cout << table         << std::endl;
  std::cout << table_cols    << std::endl;
  std::cout << table_headers << std::endl;
  std::cout << table_cols    << std::endl;

  double loss = mae.function(model.getResult(), batchTargetsTests);
  std::cout << "\nLoss: " << loss << std::endl;
}

void TitanicModel::save(std::string filename) {
  std::cout << "Saving: " << filename << std::endl;
  {
    std::ofstream ofs(filename);
    boost::archive::text_oarchive oa(ofs);
    oa << model;
  }
  std::cout << "Saved!" << std::endl;
}

Eigen::MatrixXd TitanicModel::predict(Eigen::MatrixXd input) {
  model.forward(input);
  return model.getResult();
}

TitanicModel::~TitanicModel() {
}


int main(int argc, char *argv[]) {

  TitanicModel   titanicModel = TitanicModel("../models/titanic.dat");


  titanicModel.train(128,
                     100,
                     1000,
                     0.00001);
  titanicModel.tests();

  char flag = 'n';

  std::cout << "Do you want to save model? (y/n): ";
  std::cin >> flag;

  if (flag == 'y') {
    titanicModel.save("../models/titanic.dat");
  }

  return 0;
}
