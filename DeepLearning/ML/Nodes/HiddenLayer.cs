using DeepLearning.ML.Optimizers;
using static DeepLearning.MatrixOperations;

namespace DeepLearning.ML.Nodes.HiddenLayers;

/// <summary>
/// Implementación de la capa oculta.
/// </summary>
public abstract class HiddenLayer
{
    // Número de datos de entrada
    private int _numInputs;
    // Al asignar los datos de entrada, se asigna la matriz de pesos
    public virtual int NumInputs
    {
        get => _numInputs;
        set
        {
            _numInputs = value;
            Weights = new double[_numInputs, NumOutputs];
        }
    }
    
    // Número de datos de salida
    public readonly int NumOutputs;

    // Atributos como pesos y sesgo
    protected double[,] Weights;
    private double[] _bias;
    
    // Atributos para optimización
    private double[,] _layerInputs;
    private double[,] _preActivation;
    
    // Optimizador
    private Optimizer Optimizer { get; set; }
    
    /// <summary>
    /// Constructor de Capa Oculta.
    /// </summary>
    /// <param name="numOutputs">No. de salidas.</param>
    /// <param name="optimizer">Optimizador.</param>
    protected HiddenLayer(int numOutputs, Optimizer optimizer)
    {
        // Se guardan los atributos
        NumOutputs = numOutputs;
        _bias = new double[numOutputs];
        Optimizer = optimizer;
        // Se establece el tamaño del optimizador
        Optimizer.Size = numOutputs;
    }
        
    /// <summary>
    /// Función de activación.
    /// </summary>
    /// <param name="preActivation">Valores de entrada.</param>
    /// <returns>Valor de salida.</returns>
    protected abstract double[,] ActivFunc(double[,] preActivation, bool trainingMode);
    
    /// <summary>
    /// Derivada de la función de activación.
    /// </summary>
    /// <param name="preActivation">Valores de entrada.</param>
    /// <returns>Pendiente en ese punto.</returns>
    protected abstract double[,] DerivFunc(double[,] preActivation);
    
    /// <summary>
    /// Propaga la señal hacia adelante.
    /// </summary>
    /// <param name="layerInput">Valores de entrada.</param>
    /// <returns>Valores de salida.</returns>
    public double[] ForwardPropagation(double[] layerInput)
    {
        var size = layerInput.GetLength(0);
        // Se declara e instancia una matriz bidimensional con una sola fila
        var layerInputs = new double[1, size];
        
        // Se copian los elementos del arreglo de entrada a la matriz
        for (var i = 0; i < size; i++)
        {
            layerInputs[0, i] = layerInput[i];
        }
        
        // Se regresa 
        return AverageAlongDimension(ForwardPropagation(layerInputs, false), 0);
    }
    
    /// <summary>
    /// Propaga la señal entrante hacia adelante.
    /// </summary>
    /// <param name="layerInputs">Valores de entrada (batch).</param>
    /// <param name="trainingMode">Indica si se está en modo de entrenamiento.</param>
    /// <returns>Valores de salida.</returns>
    public double[,] ForwardPropagation(double[,] layerInputs, bool trainingMode)
    {
        // Se guardan los valores de entrada para posterior optimización.
        _layerInputs = layerInputs;
        
        // Se guardan los valores pre-activación para posterior optimización.
        _preActivation = MatrixMult(layerInputs, Weights);
        
        var size = new[] { _preActivation.GetLength(0), _preActivation.GetLength(1) };
        
        // Por cada elemento
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Se suma su sesgo
                _preActivation[i, j] += _bias[j];
            }
        }
        
        // Se regresa la matriz resultante de la activación de los nodos
        return ActivFunc(_preActivation, trainingMode);
    }
    
    /// <summary>
    /// Propaga la señal hacia atrás, optimizando los pesos y sesgos.
    /// </summary>
    /// <param name="gradients">Derivada de la función de pérdida respecto a los valores de salida.</param>
    /// <returns>Derivada de la función de perdida respecto a los valore de salida de la capa anterior.</returns>
    public double[,] BackPropagation(double[,] gradients)
    {
        // Se guarda el tamaño del batch
        var batchSize = _layerInputs.GetLength(0);
        
        // Se calculan los valores de delta
        var deltas = HadamardProduct(gradients, DerivFunc(_preActivation));
        
        // Se calculan las derivadas de la función de perdida respecto a la capa anterior
        var nextGradients = MatrixMult(deltas, Transpose(Weights));
        
        // Se calculan las gradientes de los pesos
        var weightsGradients = MatrixMult(Transpose(_layerInputs), deltas);
        weightsGradients = ApplyFunc(x => x / batchSize, weightsGradients);
        
        // Se calculan las gradientes de los sesgos
        var biasGradients = AverageAlongDimension(deltas, 0);
        
        // Se aplica el optimizador a las gradientes de peso y bias
        weightsGradients = Optimizer.OptimizeWeights(weightsGradients);
        biasGradients = Optimizer.OptimizeBias(biasGradients);
        
        // Se actualizan los valores de los pesos y sesgos
        WeightUpdate(weightsGradients);
        BiasUpdate(biasGradients);
        
        // Se regresa nextGradients (función de perdida respecto a la capa anterior)
        return nextGradients;
    }

    /// <summary>
    /// Actualiza los valores de los pesos.
    /// </summary>
    /// <param name="weightsGradients">Gradientes de los pesos.</param>
    private void WeightUpdate(double[,] weightsGradients)
    {
        var size = new[] { Weights.GetLength(0), Weights.GetLength(1) };
        // Por cada peso
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Se resta su gradiente
                Weights[i, j] -= weightsGradients[i, j];
            }
        }
    }

    /// <summary>
    /// Actualiza los valores de los sesgos.
    /// </summary>
    /// <param name="biasGradients">Gradientes de los sesgos.</param>
    private void BiasUpdate(double[] biasGradients)
    {
        var size = biasGradients.GetLength(0);
        // Por cada sesgo
        for (var i = 0; i < size; i++)
        {
            // Se resta su gradiente
            _bias[i] -= biasGradients[i];
        }
    }
}