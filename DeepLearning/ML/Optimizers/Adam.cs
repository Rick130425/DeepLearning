namespace DeepLearning.ML.Optimizers;

/// <summary>
/// Optimizador Adam (Adaptive Moment Estimation).
/// Combina las ideas de Momentum y RMSprop para realizar actualizaciones adaptativas de los pesos.
/// </summary>
public class Adam : Optimizer
{
    // Tasa de aprendizaje para ajustar los pesos.
    private double _learningRate; 
    // Coeficiente para el promedio móvil del primer momento (gradiente).
    private double _beta1; 
    // Coeficiente para el promedio móvil del segundo momento (cuadrado del gradiente).
    private double _beta2; 
    // Término de suavizado para evitar la división por cero.
    private double _epsilon; 
    // Primer momento (media móvil del gradiente)
    private double[,] _m; 
    // Segundo momento (media móvil del cuadrado del gradiente)
    private double[,] _v; 
    // Primer momento para sesgos
    private double[] _mBias; 
    // Segundo momento para sesgos
    private double[] _vBias; 
    // Contador de iteraciones
    private int _t; 
    // Agrega un nuevo miembro para el Weight Decay
    private double _weightDecay;

    // Modifica el constructor para aceptar el Weight Decay como un parámetro
    public Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double weightDecay = 0.01)
    {
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay; // Inicializa el Weight Decay
        _t = 0;
    }
        
    /// <summary>
    /// Optimiza los gradientes de los pesos utilizando el algoritmo Adam.
    /// </summary>
    /// <param name="weightsGradients">Gradientes de los pesos a optimizar.</param>
    /// <returns>Gradientes de los pesos optimizados.</returns>
    public override double[,] OptimizeWeights(double[,] weightsGradients)
    {
        // Inicializa los momentos si es la primera vez
        if (_m == null)
        {
            _m = new double[weightsGradients.GetLength(0), weightsGradients.GetLength(1)];
            _v = new double[weightsGradients.GetLength(0), weightsGradients.GetLength(1)];
        }

        _t++; // Incrementa el contador de iteraciones
        var size = new[] { weightsGradients.GetLength(0), weightsGradients.GetLength(1) };

        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Actualiza los momentos
                _m[i, j] = _beta1 * _m[i, j] + (1 - _beta1) * weightsGradients[i, j];
                _v[i, j] = _beta2 * _v[i, j] + (1 - _beta2) * Math.Pow(weightsGradients[i, j], 2);

                // Corrige los momentos para sesgo inicial
                var mCorrected = _m[i, j] / (1 - Math.Pow(_beta1, _t));
                var vCorrected = _v[i, j] / (1 - Math.Pow(_beta2, _t));
                
                // Aplica el Weight Decay antes de actualizar los gradientes de los pesos
                weightsGradients[i, j] += _weightDecay * weightsGradients[i, j];

                // Actualiza los gradientes de los pesos
                weightsGradients[i, j] -= _learningRate * mCorrected / (Math.Sqrt(vCorrected) + _epsilon);
            }
        }

        return weightsGradients;
    }
        
    /// <summary>
    /// Optimiza los gradientes de los sesgos utilizando el algoritmo Adam.
    /// </summary>
    /// <param name="biasGradients">Gradientes de los sesgos a optimizar.</param>
    /// <returns>Gradientes de los sesgos optimizados.</returns>
    public override double[] OptimizeBias(double[] biasGradients)
    {
        // Inicializa los momentos para sesgos si es la primera vez
        if (_mBias == null)
        {
            _mBias = new double[biasGradients.Length];
            _vBias = new double[biasGradients.Length];
        }

        _t++; // Incrementa el contador de iteraciones
        var size = biasGradients.Length;

        for (var i = 0; i < size; i++)
        {
            // Actualiza los momentos para sesgos
            _mBias[i] = _beta1 * _mBias[i] + (1 - _beta1) * biasGradients[i];
            _vBias[i] = _beta2 * _vBias[i] + (1 - _beta2) * Math.Pow(biasGradients[i], 2);

            // Corrige los momentos para sesgo inicial
            var mCorrected = _mBias[i] / (1 - Math.Pow(_beta1, _t));
            var vCorrected = _vBias[i] / (1 - Math.Pow(_beta2, _t));

            // Actualiza los gradientes de los sesgos
            biasGradients[i] -= _learningRate * mCorrected / (Math.Sqrt(vCorrected) + _epsilon);
        }

        return biasGradients;
    }
}