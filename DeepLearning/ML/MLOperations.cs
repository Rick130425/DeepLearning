using ClosedXML.Excel;
using System.Collections.Generic;
using System.Globalization;

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

    /// <summary>
    /// Lee datos de un archivo CSV y los convierte en un dataset.
    /// </summary>
    /// <param name="filePath">Ruta del archivo CSV.</param>
    /// <param name="inputColumns">Columna de inicio y fin de los datos de entrada.</param>
    /// <param name="outputColumn">Columna de los datos de salida.</param>
    /// <param name="hasHeaders">Indica si el archivo CSV tiene encabezados.</param>
    /// <returns>
    /// Dos arreglos: 
    /// 1. Un arreglo de arreglos de double con los datos de entrada.
    /// 2. Un arreglo de strings para los datos de salida.
    /// </returns>
    public static (double[][], string[]) ReadDataFromCsv(string filePath, int[] inputColumns, int outputColumn, 
        bool hasHeaders = false)
    {
        var inputData = new List<double[]>();
        var outputData = new List<string>();

        // Lee todas las líneas del archivo CSV
        var lines = File.ReadAllLines(filePath);

        // Si el archivo tiene encabezados, omite la primera línea
        var startLine = hasHeaders ? 1 : 0;

        // Itera sobre cada línea del archivo, comenzando por la línea de inicio
        for (var lineIndex = startLine; lineIndex < lines.Length; lineIndex++)
        {
            var line = lines[lineIndex];
            // Divide la línea en columnas basándose en comas
            var columns = line.Split(',');

            var inputValues = new List<double>();

            // Procesa los datos de entrada
            for (var i = inputColumns[0]; i <= inputColumns[1]; i++)
            {
                inputValues.Add(double.Parse(columns[i], CultureInfo.InvariantCulture));
            }

            // Procesa los datos de salida
            // Concatena los valores de salida en una sola cadena
            var outputValue = columns[0];

            // Añade los arreglos de entrada y el valor de salida a las listas correspondientes
            inputData.Add(inputValues.ToArray());
            outputData.Add(outputValue);
        }

        // Devuelve los datos en formatos adecuados para la red neuronal
        return (inputData.ToArray(), outputData.ToArray());
    }
    
    /// <summary>
    /// Lee datos de un archivo CSV y los convierte en un dataset.
    /// </summary>
    /// <param name="filePath">Ruta del archivo CSV.</param>
    /// <param name="inputColumns">Columna de inicio y fin de los datos de entrada.</param>
    /// <param name="outputColumns">Columna de inicio y fin de los datos de salida.</param>
    /// <param name="hasHeaders">Indica si el archivo CSV tiene encabezados.</param>
    /// <returns>Dataset de datos numéricos.</returns>
    public static double[][][] ReadNumericDataFromCsv(string filePath, int[] inputColumns, int[] outputColumns, 
        bool hasHeaders = false)
    {
        var allData = new List<double[][]>();

        // Lee todas las líneas del archivo CSV
        var lines = File.ReadAllLines(filePath);

        // Si el archivo tiene encabezados, omite la primera línea
        var startLine = hasHeaders ? 1 : 0;

        // Itera sobre cada línea del archivo, comenzando por la línea de inicio
        for (var lineIndex = startLine; lineIndex < lines.Length; lineIndex++)
        {
            var line = lines[lineIndex];
            // Divide la línea en columnas basándose en comas
            var columns = line.Split(',');

            var inputValues = new List<double>();
            var outputValues = new List<double>();

            // Procesa los datos de entrada
            for (var i = inputColumns[0]; i <= inputColumns[1]; i++)
            {
                inputValues.Add(double.Parse(columns[i], CultureInfo.InvariantCulture));
            }

            // Procesa los datos de salida
            for (var i = outputColumns[0]; i <= outputColumns[1]; i++)
            {
                outputValues.Add(double.Parse(columns[i], CultureInfo.InvariantCulture));
            }

            // Añade los arreglos de entrada y salida a la lista de todos los datos
            allData.Add(new[] { inputValues.ToArray(), outputValues.ToArray() });
        }

        // Devuelve los datos en un formato adecuado para la red neuronal
        return allData.ToArray();
    }
    
    /// <summary>
    /// Realiza la codificación one-hot de un conjunto de datos.
    /// </summary>
    /// <param name="data">Arreglo con los datos de entrada a codificar.</param>
    /// <param name="possibleValues">Arreglo con los valores posibles para la codificación one-hot.</param>
    /// <returns>Un arreglo de arreglos con las codificaciones one-hot.</returns>
    public static double[][] OneHotEncode(string[] data, string[] possibleValues)
    {
        var encodedData = new List<double[]>();

        // Crea un diccionario para mapear cada valor posible a un índice
        var valueIndexMap = possibleValues.Select((value, index) => new { value, index })
            .ToDictionary(pair => pair.value, pair => pair.index);

        foreach (var item in data)
        {
            // Crea un arreglo con la misma longitud que el número de valores posibles
            var encodedItem = new double[possibleValues.Length];

            // Encuentra el índice del valor actual en el arreglo de valores posibles
            if (valueIndexMap.TryGetValue(item, out int index))
            {
                // Establece 1 en la posición correspondiente al valor
                encodedItem[index] = 1;
            }
            else
            {
                throw new ArgumentException($"El valor '{item}' no está en la lista de valores posibles.");
            }

            // Añade el arreglo codificado a la lista de datos codificados
            encodedData.Add(encodedItem);
        }

        return encodedData.ToArray();
    }
}