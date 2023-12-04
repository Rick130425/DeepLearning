using DeepLearning.ML.Optimizers;
using static DeepLearning.MatrixOperations;

namespace DeepLearning.ML.Nodes.HiddenLayers;

/// <summary>
/// Capa oculta que implementa la función de activación Softmax.
/// Softmax es ampliamente utilizada en la capa de salida de redes neuronales para clasificación multiclase.
/// </summary>
public class Softmax : HiddenLayer
{
    // Sobrescribe el setter de NumInputs para inicializar los pesos.
    public override int NumInputs
    {
        set
        {
            base.NumInputs = value;
            var random = new Random();
            // Límite para la inicialización de Xavier
            var limit = Math.Sqrt(6.0 / (NumInputs + NumOutputs)); 

            // Inicializa los pesos con valores aleatorios dentro del límite
            Weights = ApplyFunc(x => random.NextDouble() * 2 * limit - limit, Weights);
        }
    }
        
    public Softmax(int numOutputs, Optimizer optimizer) 
        : base(numOutputs, optimizer)
    {
    }

    /// <summary>
    /// Función de activación Softmax.
    /// Esta función convierte los valores de pre-activación en un vector de probabilidades.
    /// Cada valor de salida representa la probabilidad de que la entrada pertenezca a una clase particular.
    /// </summary>
    /// <param name="preActivation">Valores de pre-activación de la capa.</param>
    /// <param name="trainingMode">Indica si la red está en modo de entrenamiento.</param>
    /// <returns>Valores de activación Softmax.</returns>
    protected override double[,] ActivFunc(double[,] preActivation, bool trainingMode)
    {
        var size = new int[] { preActivation.GetLength(0), preActivation.GetLength(1) };
        var softmax = new double[size[0], size[1]];
        for (int i = 0; i < size[0]; i++)
        {
            var sumExp = 0.0;
            // Primero, calcula la suma de los exponentes de los valores de pre-activación.
            // Esto se hace para normalizar los valores y convertirlos en probabilidades.
            for (var j = 0; j < size[1]; j++)
            {
                sumExp += Math.Exp(preActivation[i, j]);
            }
            // Luego, divide cada exponencial de pre-activación por la suma total de exponentes.
            // Esto asegura que la suma de las probabilidades de salida sea igual a 1.
            for (int j = 0; j < size[1]; j++)
            {
                softmax[i, j] = Math.Exp(preActivation[i, j]) / sumExp;
            }
        }
        return softmax;
    }

    /// <summary>
    /// Derivada de la función de activación Softmax para clasificación multiclase.
    /// Esta derivada es importante para el proceso de retropropagación en el entrenamiento de la red.
    /// La derivada de Softmax es una matriz jacobiana que contiene todas las derivadas parciales.
    /// </summary>
    /// <param name="preActivation">Valores de pre-activación de la capa.</param>
    /// <returns>Derivada de la función Softmax.</returns>
    protected override double[,] DerivFunc(double[,] preActivation)
    {
        var softmax = ActivFunc(preActivation, false);
        var size = new int[] { softmax.GetLength(0), softmax.GetLength(1) };
        var deriv = new double[size[0], size[1]];

        // Calcula la derivada de Softmax para cada elemento del vector de salida.
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                for (var k = 0; k < size[1]; k++)
                {
                    // La derivada de Softmax varía dependiendo de si estamos calculando la derivada
                    // con respecto a la misma salida (caso diagonal) o con respecto a una salida diferente (caso no diagonal).
                    if (j == k)
                    {
                        // En la diagonal, la derivada es el producto de la probabilidad de la clase
                        // por la diferencia entre 1 y esa misma probabilidad.
                        deriv[i, j] += softmax[i, j] * (1 - softmax[i, k]);
                    }
                    else
                    {
                        // Fuera de la diagonal, la derivada es el producto negativo de las probabilidades
                        // de las dos clases diferentes.
                        deriv[i, j] -= softmax[i, j] * softmax[i, k];
                    }
                }
            }
        }
        return deriv;
    }
}