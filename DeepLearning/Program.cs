using DeepLearning.ML;
using DeepLearning.ML.LossFunctions;
using DeepLearning.ML.Nodes.HiddenLayers;
using DeepLearning.ML.Optimizers;

var neuralNetwork = new ML(4, new MeanSquaredError());

neuralNetwork.AddLayer(new ReLU(4, new Adam()));
neuralNetwork.AddLayer(new Sigmoid(1, new Adam()));

var data = new double[16][][];
for (var a = 0; a < 2; a++)
{
    for (var b = 0; b < 2; b++)
    {
        for (var c = 0; c < 2; c++)
        {
            for (var d = 0; d < 2; d++)
            {
                var row = (int)(a + 2 * b + Math.Pow(2, 2) * c + Math.Pow(2, 3) * d);
                data[row] = new[]
                {
                    new double[] { a, b, c, d },
                    new double[] { ((a & b) | (c & d)) & ((a == 1 ? 0 : 1) | d) & c & b }
                };
            }
        }
    }
}

Console.WriteLine("Loss before training: " + neuralNetwork.AverageLoss(data));

neuralNetwork.Train(data, 16, 1000000);

Console.WriteLine("Loss after training: " + neuralNetwork.AverageLoss(data));