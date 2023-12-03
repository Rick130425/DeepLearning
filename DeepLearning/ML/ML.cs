using DeepLearning.ML.LossFunctions;
using DeepLearning.ML.Nodes.HiddenLayers;
using static DeepLearning.ML.MLOperations;

namespace DeepLearning.ML;

/// <summary>
/// Implementación de la estructura general de red neuronal.
/// </summary>
public class ML
{
    // Número de entradas
    private int _numInputs;
    // Lista de capas ocultas
    private readonly List<HiddenLayer> _hiddenLayers;
    // Función de pérdida
    public LossFunction Loss;
    
    /// <summary>
    /// Constructor de ML.
    /// </summary>
    /// <param name="numInputs">Número de datos de entrada.</param>
    /// <param name="loss">Función de pérdida.</param>
    public ML(int numInputs, LossFunction loss)
    {
        // Guarda los datos de entrada
        _numInputs = numInputs;
        // Inicializa la lista de capas ocultas
        _hiddenLayers = new List<HiddenLayer>();
        // Y guarda la función de pérdida
        Loss = loss;
    }
    
    /// <summary>
    /// Agrega capas a la red neuronal.
    /// </summary>
    /// <param name="layer">Capa a agregar.</param>
    public void AddLayer(HiddenLayer layer)
    {
        // Si no hay capas ocultas aún
        if (_hiddenLayers.Count == 0)
        {
            // Se establece que la capa oculta tendrá la misma cantidad de entradas que la red neuronal
            layer.NumInputs = _numInputs;
        }
        // Si no
        else
        {
            // Se guarda el número de capas
            var numLayers = _hiddenLayers.Count;
            // Y la cantidad de entrada de la capa será igual al número de salidas de la última capa
            layer.NumInputs = _hiddenLayers[numLayers - 1].NumOutputs;
        }
        // Por último, se agrega la nueva capa
        _hiddenLayers.Add(layer);
    }
    
    /// <summary>
    /// Propaga los datos de entrada a través de la red neuronal.
    /// </summary>
    /// <param name="inputs">Datos de entrada.</param>
    /// <returns>Datos de salida.</returns>
    public double[] ForwardPropagation(double[] inputs)
    {
        // Propaga los datos por todas las capas ocultas, siendo la salida de una capa la entrada de la siguiente
        return _hiddenLayers.Aggregate(inputs, (current, layer) => layer.ForwardPropagation(current));
    }
    
    /// <summary>
    /// Propaga un batch de entrada a través de la red neuronal.
    /// </summary>
    /// <param name="inputs">Batch de entrada.</param>
    /// <returns>Batch de salida.</returns>
    public double[,] ForwardPropagation(double[,] inputs)
    {
        // Propaga el batch por todas las capas ocultas, siendo la salida de una capa la entrada de la siguiente
        return _hiddenLayers.Aggregate(inputs, (current, layer) => layer.ForwardPropagation(current));
    }
    
    /// <summary>
    /// Propaga el error de predicciones hacia atrás, ajustando los parámetros.
    /// </summary>
    /// <param name="gradients">Derivada de la función de error respecto a las salidas.</param>
    private void BackPropagation(double[,] gradients)
    {
        // Se guarda la cantidad de capas ocultas
        var size = _hiddenLayers.Count;
        // Se guardan las gradientes en una variable llamada gradiente
        var gradient = gradients;
        // Desde adelante hacia atrás, se propaga la gradiente a través de las capas
        for (var i = size - 1; i >= 0; i--)
        {
            gradient = _hiddenLayers[i].BackPropagation(gradient);
        }
    }
    
    /// <summary>
    /// Entrena la red neuronal.
    /// </summary>
    /// <param name="data">Arreglo 3D de datos (Arreglo de arreglo de arreglos).</param>
    /// <param name="batchSize">Tamaño de batch.</param>
    /// <param name="epochs">Número de épocas de entrenamiento.</param>
    /// <param name="truncate">Truncar los datos si no se completan los batches.</param>
    /// <returns>Error de predicción época.</returns>
    public double[] Train(double[][][] data, int batchSize, int epochs, bool truncate = true)
    {
        // Se declara e inicializa un arreglo de pérdida
        var lossPerEpoch = new double[epochs];
        // Por cada época
        for (var i = 0; i < epochs; i++)
        {
            // Se cambia aleatoriamente el orden de los datos
            data = RandomizeDataOrder(data);
            // Se dividen los datos en batches
            var batches = Batching(data, batchSize, truncate);
            // Se guarda la cantidad de batches
            var numBatches = batches.Length;
            // Por cada batch
            for (var j = 0; j < numBatches; j++)
            {
                // Se calculan sus valores de salida
                var outputValues = ForwardPropagation(batches[j][0]);
                // Se calcula su error de predicción
                lossPerEpoch[i] += Loss.CalcLoss(outputValues, batches[j][1]);
                // Y se propaga el error a través de la red neuronal
                BackPropagation(Loss.DerivLoss(outputValues, batches[j][1]));
            }
            
            // Se divide el error entre el número de batches para promediar
            lossPerEpoch[i] /= numBatches;
        }
        
        // Por último, se regresa el arreglo de pérdida
        return lossPerEpoch;
    }
    
    /// <summary>
    /// Calcula el error promedio de la red neuronal en un set de datos.
    /// </summary>
    /// <param name="data">Set de datos.</param>
    /// <returns>Error promedio.</returns>
    public double AverageLoss(double[][][] data)
    {
        // Se declara la variable de pérdida y se inicializa con 0.0
        var loss = 0.0;
        // Se divide el set de datos en batches de un elemento
        var batches = Batching(data, 1);
        // Se guarda la cantidad de batches
        var numBatches = batches.Length;
        // Por cada batch
        for (var j = 0; j < numBatches; j++)
        {
            // Se calculan los valores de salida
            var outputValues = ForwardPropagation(batches[j][0]);
            // Se calcula el error de predicción
            loss += Loss.CalcLoss(outputValues, batches[j][1]);
        }
        // Se divide el error entre el número de batches para promediar
        loss /= numBatches;
        
        // Por último, se regresa el error promedio.
        return loss;
    }
}