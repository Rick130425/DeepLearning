using DeepLearning.ML.Optimizers;

namespace DeepLearning.ML.Nodes.HiddenLayers
{
    /// <summary>
    /// Capa oculta que implementa Dropout.
    /// </summary>
    public class Dropout : HiddenLayer
    {
        // Tasa de Dropout
        private readonly double _dropoutRate;
        // Generador de números aleatorios
        private readonly Random _random;
        // Máscara de dropout
        private bool[] _dropoutMask;

        /// <summary>
        /// Constructor de la clase Dropout.
        /// Inicializa la tasa de Dropout y el generador de números aleatorios.
        /// </summary>
        /// <param name="numOutputs">Número de salidas de la capa.</param>
        /// <param name="optimizer">Optimizador a utilizar.</param>
        /// <param name="dropoutRate">Tasa de Dropout (0 a 1).</param>
        public Dropout(int numOutputs, Optimizer optimizer, double dropoutRate) 
            : base(numOutputs, optimizer)
        {
            _dropoutRate = dropoutRate;
            _random = new Random();
        }

        /// <summary>
        /// Función de activación aplicada a la capa.
        /// Durante el entrenamiento, aplica Dropout a los nodos de la capa.
        /// </summary>
        /// <param name="preActivation">Valores de pre-activación de la capa.</param>
        /// <param name="trainingMode">Indica si la red está en modo de entrenamiento.</param>
        /// <returns>Valores de activación con Dropout aplicado.</returns>
        protected override double[,] ActivFunc(double[,] preActivation, bool trainingMode)
        {
            // Aplica Dropout solo durante el entrenamiento
            if (trainingMode)
            {
                var size = preActivation.GetLength(1);
                _dropoutMask = new bool[size];
                for (var i = 0; i < size; i++)
                {
                    // Determina aleatoriamente si el nodo se "apaga"
                    _dropoutMask[i] = _random.NextDouble() > _dropoutRate;
                    if (_dropoutMask[i])
                    {
                        // Aplica el efecto de Dropout a los nodos activos
                        for (var j = 0; j < preActivation.GetLength(0); j++)
                        {
                            preActivation[j, i] *= _dropoutMask[i] ? 1.0 : 0.0;
                        }
                    }
                }
            }
            return preActivation;
        }

        /// <summary>
        /// Derivada de la función de activación.
        /// Durante el entrenamiento, aplica la máscara de Dropout a la derivada.
        /// </summary>
        /// <param name="preActivation">Valores de pre-activación de la capa.</param>
        /// <returns>Derivada de la función de activación con Dropout aplicado.</returns>
        protected override double[,] DerivFunc(double[,] preActivation)
        {
            var size = new[]{ preActivation.GetLength(0), preActivation.GetLength(1) };
            // Aplica la máscara de Dropout a la derivada
            for (var i = 0; i < size[0]; i++)
            {
                if (!_dropoutMask[i])
                {
                    // "Apaga" los nodos que fueron desactivados durante la activación
                    for (var j = 0; j < size[1]; j++)
                    {
                        preActivation[j, i] = 0.0;
                    }
                }
            }
            return preActivation;
        }
    }
}