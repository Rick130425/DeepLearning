using DeepLearning.ML.Optimizers;
using static DeepLearning.MatrixOperations;

namespace DeepLearning.ML.Nodes.HiddenLayers;

/// <summary>
/// Capa oculta que implementa la función de activación Sigmoid.
/// </summary>
public class Sigmoid : HiddenLayer
{
    // Sobrescribe el setter de NumInputs para inicializar los pesos.
    public override int NumInputs
    {
        set
        {
            base.NumInputs = value;
            var random = new Random();
            // Se establecen los límites de generación de los números (inicialización de Xavier)
            var limit = 1 / Math.Sqrt(value); 

            // Inicializa los pesos con valores aleatorios dentro del límite
            Weights = ApplyFunc(x => random.NextDouble() * 2 * limit - limit, Weights);
        }
    }
        
    /// <summary>
    /// Constructor de la clase Sigmoid.
    /// </summary>
    /// <param name="numOutputs">Número de salidas de la capa.</param>
    /// <param name="optimizer">Optimizador a utilizar.</param>
    public Sigmoid(int numOutputs, Optimizer optimizer) 
        : base(numOutputs, optimizer)
    {
    }

    /// <summary>
    /// Función de activación Sigmoid.
    /// Convierte los valores de pre-activación en probabilidades entre 0 y 1.
    /// </summary>
    /// <param name="preActivation">Valores de pre-activación de la capa.</param>
    /// <param name="trainingMode">Indica si la red está en modo de entrenamiento.</param>
    /// <returns>Valores de activación Sigmoid.</returns>
    protected override double[,] ActivFunc(double[,] preActivation, bool trainingMode)
    {
        // f(x) = 1 / (1 + e^x)
        return ApplyFunc(x => 1.0 / (1.0 + Math.Exp(-x)), preActivation);
    }

    /// <summary>
    /// Derivada de la función de activación Sigmoid.
    /// Es útil durante la retropropagación para calcular gradientes.
    /// </summary>
    /// <param name="preActivation">Valores de pre-activación de la capa.</param>
    /// <returns>Derivada de la función Sigmoid.</returns>
    protected override double[,] DerivFunc(double[,] preActivation)
    {
        var sigmoid = ActivFunc(preActivation, false);
        // f'(x) = f(x) * (1 - f(x))
        return ApplyFunc(x => x * (1 - x), sigmoid);
    }
}