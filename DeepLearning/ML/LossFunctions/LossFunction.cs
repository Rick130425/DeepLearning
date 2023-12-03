namespace DeepLearning.ML.LossFunctions;

/// <summary>
/// Implementación de la función de pérdida.
/// </summary>
public abstract class LossFunction
{
    /// <summary>
    /// Calcula el valor promedio de pérdida en un conjunto de datos.
    /// </summary>
    /// <param name="outputValues">Valores de salida de la red neuronal.</param>
    /// <param name="expectedValues">Valores de salida esperados.</param>
    /// <returns>Error de la red neuronal.</returns>
    public abstract double CalcLoss(double[,] outputValues, double[,] expectedValues);
    
    /// <summary>
    /// Regresa la derivada de la función de pérdida respecto a las salidas de la red neuronal.
    /// </summary>
    /// <param name="outputValues">Valores de salida de la red neuronal.</param>
    /// <param name="expectedValues">Valores de salida esperados.</param>
    /// <returns>Matriz de derivadas.</returns>
    public abstract double[,] DerivLoss(double[,] outputValues, double[,] expectedValues);
}