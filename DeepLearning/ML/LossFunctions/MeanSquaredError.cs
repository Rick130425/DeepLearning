namespace DeepLearning.ML.LossFunctions;

/// <summary>
/// Función de pérdida Mean Squared Error.
/// </summary>
public class MeanSquaredError : LossFunction
{
    /// <summary>
    /// Calcula el valor promedio de pérdida en un conjunto de datos utilizando MSE.
    /// </summary>
    /// <param name="outputValues">Valores de salida de la red neuronal.</param>
    /// <param name="expectedValues">Valores de salida esperados.</param>
    /// <returns>Error de la red neuronal.</returns>
    public override double CalcLoss(double[,] outputValues, double[,] expectedValues)
    {
        var size = new[] { outputValues.GetLength(0), outputValues.GetLength(1) };
        var batchSize = outputValues.GetLength(0);
        // Declara y asigna la pérdida como 0.0
        var loss = 0.0;
        
        // Por cada valor de salida
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Se calcula su error y se suma el valor de perdida
                loss += Math.Pow(expectedValues[i, j] - outputValues[i, j], 2);
            }
        }
        
        // Por último se promedia, dividiéndolo entre el tamaño del batch
        loss /= batchSize;
        // Y se regresa la pérdida
        return loss;
    }
    
    /// <summary>
    /// Regresa la derivada de la función de pérdida respecto a las salidas de la red neuronal (MSE).
    /// </summary>
    /// <param name="outputValues">Valores de salida de la red neuronal.</param>
    /// <param name="expectedValues">Valores de salida esperados.</param>
    /// <returns>Matriz de derivadas.</returns>
    public override double[,] DerivLoss(double[,] outputValues, double[,] expectedValues)
    {
        var size = new int[] { outputValues.GetLength(0), outputValues.GetLength(1) };
        // Se declara la matriz de gradientes
        var gradients = new double[size[0], size[1]];
        
        // Por cada posición de la matriz de gradientes
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Se calcula la gradiente
                gradients[i, j] = 2 * (outputValues[i, j] - expectedValues[i, j]);
            }
        }
        
        // Regresa la matriz de gradientes
        return gradients;
    }
}