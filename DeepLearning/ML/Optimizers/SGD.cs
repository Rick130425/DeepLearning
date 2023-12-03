using DeepLearning.ML.Nodes.HiddenLayers;
using static DeepLearning.MatrixOperations;

namespace DeepLearning.ML.Optimizers;

/// <summary>
/// Optimizador que implementa el algoritmo SGD.
/// </summary>
public class SGD : Optimizer
{
    // Tasa de aprendizaje
    public double LearningRate { get; set; }

    /// <summary>
    /// Constructor del optimizador (Descenso de Gradiente Estocástico)
    /// </summary>
    /// <param name="learningRate">Tasa de aprendizaje</param>
    public SGD(double learningRate)
    {
        LearningRate = learningRate;
    }
    
    public override double[,] OptimizeWeights(double[,] weightsGradients)
    {
        var size = new[] { weightsGradients.GetLength(0), weightsGradients.GetLength(1) };
        // Por cada gradiente de peso
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Se aplica una función de actualización
                weightsGradients[i, j] = UpdateFunction(weightsGradients[i, j], LearningRate);
            }
        }
        
        // Se regresan las gradientes optimizadas.
        return weightsGradients;
    }
    
    public override double[] OptimizeBias(double[] biasGradients)
    {
        var size = biasGradients.Length;
        // Por cada gradiente de sesgo
        for (var i = 0; i < size; i++)
        {
            // Se aplica una función de actualización
            biasGradients[i] = UpdateFunction(biasGradients[i], LearningRate);
        }
        
        // Se regresan las gradientes optimizadas.
        return biasGradients;
    }
    
    /// <summary>
    /// Función de actualización de gradiente.
    /// </summary>
    /// <param name="gradient">Gradiente.</param>
    /// <param name="learningRate">Tasa de aprendizaje.</param>
    /// <returns>Gradiente optimizada.</returns>
    private double UpdateFunction(double gradient, double learningRate)
    {
        return learningRate * gradient;
    }
}