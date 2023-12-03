namespace DeepLearning.ML.Optimizers;

/// <summary>
/// Implementación de algoritmo de optimización.
/// </summary>
public abstract class Optimizer
{
    // Tamaño del optimizador
    public virtual int Size { get; set; }

    /// <summary>
    /// Optimizador de gradientes de peso.
    /// </summary>
    /// <param name="weightsGradients">Gradientes de los pesos.</param>
    /// <returns>Gradientes después de optimización.</returns>
    public abstract double[,] OptimizeWeights(double[,] weightsGradients);
    
    /// <summary>
    /// Optimizador de gradientes de sesgo.
    /// </summary>
    /// <param name="biasGradients">Gradientes de los sesgos.</param>
    /// <returns>Gradientes después de optimización.</returns>
    public abstract double[] OptimizeBias(double[] biasGradients);
    
}