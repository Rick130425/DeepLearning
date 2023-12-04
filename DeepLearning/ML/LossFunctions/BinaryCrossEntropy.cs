﻿namespace DeepLearning.ML.LossFunctions;

/// <summary>
/// Función de pérdida de Entropía Cruzada Binaria.
/// Utilizada en problemas de clasificación binaria.
/// Mide el desempeño de un modelo cuya salida es una probabilidad entre 0 y 1.
/// </summary>
public class BinaryCrossEntropy : LossFunction
{
    /// <summary>
    /// Calcula el valor promedio de pérdida en un conjunto de datos utilizando la Entropía Cruzada Binaria.
    /// </summary>
    /// <param name="outputValues">Valores de salida de la red neuronal (probabilidades).</param>
    /// <param name="expectedValues">Valores de salida esperados (etiquetas binarias).</param>
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
                // Fórmula de la Entropía Cruzada Binaria
                loss -= expectedValues[i, j] * Math.Log(outputValues[i, j]) + 
                        (1 - expectedValues[i, j]) * Math.Log(1 - outputValues[i, j]);
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
        var size = new[] { outputValues.GetLength(0), outputValues.GetLength(1) };
        var gradients = new double[size[0], size[1]];

        // Calcula el gradiente para cada salida
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Derivada de la Entropía Cruzada Binaria
                gradients[i, j] = (outputValues[i, j] - 
                                   expectedValues[i, j]) / (outputValues[i, j] * (1 - outputValues[i, j]));
            }
        }

        return gradients;
    }
}