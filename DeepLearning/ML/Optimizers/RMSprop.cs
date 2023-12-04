namespace DeepLearning.ML.Optimizers;

/// <summary>
/// Optimizador RMSprop (Root Mean Square Propagation).
/// Ajusta la tasa de aprendizaje para cada peso individualmente.
/// Utiliza un promedio móvil del cuadrado del gradiente para realizar esta adaptación.
/// </summary>
public class RMSprop : Optimizer
{
    // Tasa de aprendizaje para ajustar los pesos.
    private double _learningRate; 
    // Tasa de decaimiento para el promedio móvil del cuadrado del gradiente.
    private double _decayRate; 
    // Término de suavizado para evitar la división por cero.
    private double _epsilon; 
    // Cache para almacenar el promedio móvil del cuadrado del gradiente.
    private double[,] _cache; 
    // Cache para almacenar el promedio móvil del cuadrado del gradiente de sesgos. 
    private double[] _cacheBias; 

    public RMSprop(double learningRate = 0.001, double decayRate = 0.9, double epsilon = 1e-8)
    {
        this._learningRate = learningRate;
        this._decayRate = decayRate;
        this._epsilon = epsilon;
    }
    
    /// <summary>
    /// Optimiza los gradientes de los pesos utilizando el algoritmo RMSprop.
    /// </summary>
    /// <param name="weightsGradients">Gradientes de los pesos a optimizar.</param>
    /// <returns>Gradientes de los pesos optimizados.</returns>
    public override double[,] OptimizeWeights(double[,] weightsGradients)
    {
        // Inicializa el cache si es la primera vez
        if (_cache == null)
        {
            _cache = new double[weightsGradients.GetLength(0), weightsGradients.GetLength(1)];
        }

        var size = new[] { weightsGradients.GetLength(0), weightsGradients.GetLength(1) };

        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Actualiza el cache con el promedio móvil del cuadrado del gradiente
                _cache[i, j] = _decayRate * _cache[i, j] + (1 - _decayRate) * Math.Pow(weightsGradients[i, j], 2);

                // Ajusta los gradientes de los pesos
                weightsGradients[i, j] -= _learningRate * weightsGradients[i, j] / (Math.Sqrt(_cache[i, j]) + _epsilon);
            }
        }

        return weightsGradients;
    }
    
    /// <summary>
    /// Optimiza los gradientes de los sesgos utilizando el algoritmo RMSprop.
    /// </summary>
    /// <param name="biasGradients">Gradientes de los sesgos a optimizar.</param>
    /// <returns>Gradientes de los sesgos optimizados.</returns>
    public override double[] OptimizeBias(double[] biasGradients)
    {
        // Inicializa el cache para sesgos si es la primera vez
        if (_cacheBias == null)
        {
            _cacheBias = new double[biasGradients.Length];
        }

        var size = biasGradients.Length;

        for (var i = 0; i < size; i++)
        {
            // Actualiza el cache para sesgos con el promedio móvil del cuadrado del gradiente
            _cacheBias[i] = _decayRate * _cacheBias[i] + (1 - _decayRate) * Math.Pow(biasGradients[i], 2);

            // Ajusta los gradientes de los sesgos
            biasGradients[i] -= _learningRate * biasGradients[i] / (Math.Sqrt(_cacheBias[i]) + _epsilon);
        }

        return biasGradients;
    }
}