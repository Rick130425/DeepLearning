using DeepLearning.ML.Optimizers;

namespace DeepLearning.ML.Nodes.HiddenLayers;

/// <summary>
/// Capa oculta que implementa Dropout.
/// </summary>
public class Dropout : HiddenLayer
{
    private readonly double _dropoutRate;
    private Random _random;
    private bool[] _dropoutMask;

    /// <summary>
    /// Constructor de Dropout.
    /// </summary>
    /// <param name="numOutputs">Número de salidas.</param>
    /// <param name="optimizer">Optimizador.</param>
    /// <param name="dropoutRate">Tasa de Dropout (0 a 1).</param>
    public Dropout(int numOutputs, Optimizer optimizer, double dropoutRate) 
        : base(numOutputs, optimizer)
    {
        _dropoutRate = dropoutRate;
        _random = new Random();
    }

    protected override double[,] ActivFunc(double[,] preActivation, bool trainingMode)
    {
        // Durante el entrenamiento, aplica Dropout
        if (trainingMode)
        {
            var size = preActivation.GetLength(1);
            _dropoutMask = new bool[size];
            for (var i = 0; i < size; i++)
            {
                _dropoutMask[i] = _random.NextDouble() > _dropoutRate;
                if (_dropoutMask[i])
                {
                    for (var j = 0; j < preActivation.GetLength(0); j++)
                    {
                        preActivation[j, i] *= _dropoutMask[i] ? 1.0 : 0.0;
                    }
                }
            }
        }
        return preActivation;
    }

    protected override double[,] DerivFunc(double[,] preActivation)
    {
        // Durante el entrenamiento, aplica la máscara de Dropout
        for (var i = 0; i < preActivation.GetLength(1); i++)
        {
            if (!_dropoutMask[i])
            {
                for (var j = 0; j < preActivation.GetLength(0); j++)
                {
                    preActivation[j, i] = 0.0;
                }
            }
        }
        return preActivation;
    }
}