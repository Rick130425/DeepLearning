namespace DeepLearning.ML;

public static class MLOperations
{
    /// <summary>
    /// Divide un set de datos en batches.
    /// </summary>
    /// <param name="data">Set de datos.</param>
    /// <param name="batchSize">Tamaño de batches.</param>
    /// <param name="truncate">Truncar los datos si no se completan los batches.</param>
    /// <returns>Datos en forma de batches.</returns>
    public static double[][][,] Batching(double[][][] data, int batchSize, bool truncate = true)
    {
        // Cantidad de datos (conjuntos de entrada y salida)
        var dataSize = data.GetLength(0);
        // Cantidad de datos de entrada y salida
        var ioSize = new[] { data[0][0].Length, data[0][1].Length };
        
        // Determina si habrá batches incompletos
        var incompleteBatch = dataSize % batchSize != 0 && !truncate;
        // Número de batches (sin contar el incompleto)
        var numBatches = dataSize / batchSize;
        // Se declara el arreglo de batches
        var batches = new double[numBatches + (incompleteBatch ? 1 : 0)][][,];
        
        // Por cada batch
        for (var i = 0; i < numBatches; i++)
        {
            // Se declara el arreglo para entradas y salidas
            batches[i] = new double[2][,];
            for (var j = 0; j < 2; j++)
            {
                // Se declara la matriz de entrada o salida
                batches[i][j] = new double[batchSize, ioSize[j]];
                for (var k = 0; k < batchSize; k++)
                {
                    // Se calcula la fila en que corresponde en los datos
                    var row = i * batchSize + k;
                    // Y se copian los datos correspondientes en el batch
                    for (var w = 0; w < ioSize[j]; w++)
                    {
                        batches[i][j][k, w] = data[row][j][w];
                    }
                }
            }
        }
        
        // Si se completan los batches, se regresan
        if (!incompleteBatch) return batches;
        
        // Si no, se calcula el nuevo tamaño del batch
        var newBatchSize = dataSize % batchSize;
        // Se declara el arreglo de entradas y salidas
        batches[numBatches] = new double[2][,];
        for (var i = 0; i < 2; i++)
        {
            // Se declara la matriz de entrada o salida
            batches[numBatches][i] = new double[newBatchSize, ioSize[i]];
            for (var j = 0; j < newBatchSize; j++)
            {
                // Se calcula la fila en que corresponde en los datos
                var row = numBatches * batchSize + j;
                // Y se copian los datos correspondientes en el batch
                for (var k = 0; k < ioSize[i]; k++)
                {
                    batches[numBatches][i][j, k] = data[row][i][k];
                }
            }
        }
        
        // Por último, se regresa el arreglo de batches
        return batches;
    }
    
    /// <summary>
    /// Cambia la posición de los conjuntos de un set de datos usando el algoritmo de Fisher-Yates.
    /// </summary>
    /// <param name="data">Set de datos.</param>
    /// <returns>Set de datos con orden aleatorio.</returns>
    public static double[][][] RandomizeDataOrder(double[][][] data)
    {
        // Se inicializa un generados de números aleatorios
        var random = new Random();
        var size = new[] { data.Length, data[0].Length };
        // Desde el último hasta el primer elemento
        for (var i = size[0] - 1; i >= 0; i--)
        {
            // Se genera un valor aleatorio entre 0 y la posición actuál
            var newRow = random.Next(i + 1);
            // Y se intercambia el elemento actual con el de esa posición
            for (var j = 0; j < size[1]; j++)
            {
                (data[i][j], data[newRow][j]) = (data[newRow][j], data[i][j]);
            }
        }
        
        // Por último, se regresa el set de datos con orden aleatorio.
        return data;
    }
}