using DeepLearning.ML.Optimizers;
using static DeepLearning.MatrixOperations;

namespace DeepLearning.ML.Nodes.HiddenLayers;

/// <summary>
/// Capa oculta que implementa la función ReLU.
/// </summary>
public class ReLU : HiddenLayer
{
    // Se sobre-escribe el comportamiento de asignación, para agregar la inicialización de valores de los pesos
    public override int NumInputs
    {
        set
        {
            base.NumInputs = value;
            // Se inicializa un generador de números aleatorios
            var random = new Random();
            // Se establecen los límites de generación de los números
            var limit = Math.Sqrt(2.0 / NumInputs);
            
            // Y se llena la matriz de pesos con valores aleatorios dentro del límite de generación
            Weights = ApplyFunc(x => random.NextDouble() * 2 * limit - limit, Weights);
        }
    }
    
    /// <summary>
    /// Constructor de ReLU.
    /// </summary>
    /// <param name="numOutputs">No. de salidas.</param>
    /// <param name="optimizer">Optimizador.</param>
    public ReLU(int numOutputs, Optimizer optimizer) : base(numOutputs, optimizer)
    {
    }
    
    /// <summary>
    /// Función de activación.
    /// </summary>
    /// <param name="preActivation">Valores de entrada.</param>
    /// <param name="trainingMode">No usada en esta función.</param>
    /// <returns>Valor de salida.</returns>
    protected override double[,] ActivFunc(double[,] preActivation, bool trainingMode)
    {
        // Si x es menor a 0, regresa 0, si no x
        return ApplyFunc(x => x < 0 ? 0 : x, preActivation);
    }
    
    /// <summary>
    /// Derivada de la función de activación.
    /// </summary>
    /// <param name="preActivation">Valores de entrada.</param>
    /// <returns>Pendiente en ese punto.</returns>
    protected override double[,] DerivFunc(double[,] preActivation)
    {
        // Si x es menor a 0, regresa 0, si no 1
        return ApplyFunc(x => x < 0 ? 0 : 1, preActivation);
    }
}