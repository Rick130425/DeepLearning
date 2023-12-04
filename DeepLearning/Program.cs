using DeepLearning.ML;
using DeepLearning.ML.LossFunctions;
using DeepLearning.ML.Nodes.HiddenLayers;
using DeepLearning.ML.Optimizers;
using static DeepLearning.ML.MLOperations;

var neuralNetwork = new ML(3, new MeanSquaredError());

neuralNetwork.AddLayer(new ReLU(8, new Adam()));
neuralNetwork.AddLayer(new ReLU(4, new Adam()));
neuralNetwork.AddLayer(new ReLU(1, new Adam()));

var data = ReadNumericDataFromCsv("./Datasets/housing.csv", new[] { 0, 2 }, new[] {3, 3}, 
    hasHeaders: true);

Console.WriteLine("Loss before training: " + neuralNetwork.AverageLoss(data));

for (var i = 0; i < 100; i++)
{
    neuralNetwork.Train(data, 8, 1);

    Console.WriteLine($"Loss after training ({i}): " + neuralNetwork.AverageLoss(data));
}