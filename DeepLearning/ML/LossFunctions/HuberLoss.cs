namespace DeepLearning.ML.LossFunctions;

/// <summary>
/// Clase que implementa la función de pérdida de Huber.
/// La pérdida de Huber es menos sensible a outliers en comparación con el MSE.
/// Combina elementos de MSE para errores pequeños y MAE para errores grandes.
/// </summary>
public class HuberLoss : LossFunction
{
    // Umbral para cambiar entre MSE y MAE
    private readonly double delta; 

    /// <summary>
    /// Inicializa una nueva instancia de la clase HuberLoss con un valor de delta especificado.
    /// </summary>
    /// <param name="delta">Umbral para cambiar entre MSE y MAE.</param>
    public HuberLoss(double delta = 1)
    {
        this.delta = delta;
    }

    /// <summary>
    /// Calcula el valor promedio de pérdida en un conjunto de datos utilizando la pérdida de Huber.
    /// </summary>
    /// <param name="outputValues">Valores de salida de la red neuronal.</param>
    /// <param name="expectedValues">Valores de salida esperados.</param>
    /// <returns>Error promedio de la red neuronal.</returns>
    public override double CalcLoss(double[,] outputValues, double[,] expectedValues)
    {
        var size = new[] { outputValues.GetLength(0), outputValues.GetLength(1) };
        var loss = 0.0;

        // Itera sobre cada elemento en el conjunto de datos
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                var error = expectedValues[i, j] - outputValues[i, j];
                // Aplica una función de pérdida cuadrática para errores pequeños y lineal para errores grandes
                loss += Math.Abs(error) < delta ? 0.5 * Math.Pow(error, 2) : delta * (Math.Abs(error) - 0.5 * delta);
            }
        }

        // Promedia la pérdida sobre el tamaño del lote
        loss /= size[0];
        return loss;
    }

    /// <summary>
    /// Calcula la derivada de la función de pérdida respecto a las salidas de la red neuronal.
    /// Esta derivada es utilizada en la retropropagación para actualizar los pesos.
    /// </summary>
    /// <param name="outputValues">Valores de salida de la red neuronal.</param>
    /// <param name="expectedValues">Valores de salida esperados.</param>
    /// <returns>Matriz de derivadas.</returns>
    public override double[,] DerivLoss(double[,] outputValues, double[,] expectedValues)
    {
        var size = new[] { outputValues.GetLength(0), outputValues.GetLength(1) };
        var gradients = new double[size[0], size[1]];

        // Calcula el gradiente para cada elemento en el conjunto de datos
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                var error = outputValues[i, j] - expectedValues[i, j];
                // Derivada de Huber Loss: lineal para errores grandes y proporcional al error para errores pequeños
                gradients[i, j] = Math.Abs(error) < delta ? error : delta * Math.Sign(error);
            }
        }

        return gradients;
    }
}