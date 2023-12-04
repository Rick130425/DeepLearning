namespace DeepLearning.ML.LossFunctions;

/// <summary>
/// Función de pérdida de Entropía Cruzada Categórica.
/// Utilizada en problemas de clasificación multiclase.
/// Mide el desempeño de un modelo cuya salida es un conjunto de probabilidades.
/// </summary>
public class CategoricalCrossEntropy : LossFunction
{
    /// <summary>
    /// Calcula el valor promedio de pérdida en un conjunto de datos utilizando la Entropía Cruzada Categórica.
    /// </summary>
    /// <param name="outputValues">Valores de salida de la red neuronal (probabilidades para cada clase).</param>
    /// <param name="expectedValues">Valores de salida esperados (etiquetas en formato one-hot).</param>
    /// <returns>Error promedio de la red neuronal.</returns>
    public override double CalcLoss(double[,] outputValues, double[,] expectedValues)
    {
        var size = new[] { outputValues.GetLength(0), outputValues.GetLength(1) };
        var batchSize = size[0];
        var loss = 0.0;

        // Calcula la pérdida para cada par de salida esperada y salida de la red
        for (var i = 0; i < batchSize; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Fórmula de la Entropía Cruzada Categórica
                loss -= expectedValues[i, j] * Math.Log(outputValues[i, j]);
            }
        }

        // Promedia la pérdida sobre el tamaño del lote
        loss /= batchSize;
        return loss;
    }

    /// <summary>
    /// Calcula la derivada de la función de pérdida respecto a las salidas de la red neuronal.
    /// </summary>
    /// <param name="outputValues">Valores de salida de la red neuronal.</param>
    /// <param name="expectedValues">Valores de salida esperados.</param>
    /// <returns>Matriz de derivadas.</returns>
    public override double[,] DerivLoss(double[,] outputValues, double[,] expectedValues)
    {
        var size = new[]{ outputValues.GetLength(0), outputValues.GetLength(1) };
        var gradients = new double[size[0], size[1]];

        // Calcula el gradiente para cada salida
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Derivada de la Entropía Cruzada Categórica
                gradients[i, j] = outputValues[i, j] - expectedValues[i, j];
            }
        }

        return gradients;
    }
}