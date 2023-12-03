namespace DeepLearning;

public static class MatrixOperations
{
    /// <summary>
    /// Multiplicación de matrices.
    /// </summary>
    /// <param name="firstMatrix">Primera matriz.</param>
    /// <param name="secondMatrix">Segunda matriz.</param>
    /// <returns>Producto de las dos matrices.</returns>
    public static double[,] MatrixMult(double[,] firstMatrix, double[,] secondMatrix)
    {
        // Almacenar las dimensiones en un arreglo
        var size = new[]
        {
            firstMatrix.GetLength(0), 
            secondMatrix.GetLength(1),
            firstMatrix.GetLength(1)
        };

        // Verificar si la multiplicación es posible
        if (size[2] != secondMatrix.GetLength(0))
        {
            throw new InvalidOperationException("Las dimensiones de las matrices no son compatibles para la multiplicación.");
        }

        var resultMatrix = new double[size[0], size[1]];

        // Por cada elemento de la nueva matriz
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                for (var k = 0; k < size[2]; k++)
                {
                    // Se calcula la suma del producto de las dos matrices
                    resultMatrix[i, j] += firstMatrix[i, k] * secondMatrix[k, j];
                }
            }
        }

        // Se regresa la nueva matriz
        return resultMatrix;
    }
    
    /// <summary>
    /// Producto de Hadamard.
    /// </summary>
    /// <param name="firstMatrix">Primera matriz.</param>
    /// <param name="secondMatrix">Segunda matriz.</param>
    /// <returns>Producto de Hadamard de las dos matrices.</returns>
    public static double[,] HadamardProduct(double[,] firstMatrix, double[,] secondMatrix)
    {
        var size = new int[] { firstMatrix.GetLength(0), firstMatrix.GetLength(1) };
        // Por cada elemento
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Se multiplican las dos matrices
                firstMatrix[i, j] *= secondMatrix[i, j];
            }
        }
        
        // Y se regresa la matriz resultante
        return firstMatrix;
    }
    
    /// <summary>
    /// Aplica una función a cada elemento de una matriz.
    /// </summary>
    /// <param name="func">Función a aplicar.</param>
    /// <param name="matrix">Matriz.</param>
    /// <returns>Matriz posterior a la aplicación de la función.</returns>
    public static double[,] ApplyFunc(Func<double, double> func, double[,] matrix)
    {
        var size = new[] { matrix.GetLength(0), matrix.GetLength(1) };
        // Por cada elemento
        for (var i = 0; i < size[0]; i++)
        {
            for (var j = 0; j < size[1]; j++)
            {
                // Se aplica la función
                matrix[i, j] = func(matrix[i, j]);
            }
        }
        
        // Y se regresa la matriz resultante
        return matrix;
    }
    
    /// <summary>
    /// Transpone una matriz.
    /// </summary>
    /// <param name="matrix">Matriz.</param>
    /// <returns>Matriz transpuesta.</returns>
    public static double[,] Transpose(double[,] matrix)
    {
        var size = new[] { matrix.GetLength(0), matrix.GetLength(1) };
        // Se declara y asigna una nueva matriz con las dimensiones transpuestas
        var transposeMatrix = new double[size[1], size[0]];
        // Por cada elemento
        for (var i = 0; i < size[1]; i++)
        {
            for (var j = 0; j < size[0]; j++)
            {
                // Se guarda en la matriz transpuesta con sus coordenadas invertidas
                transposeMatrix[i, j] = matrix[j, i];
            }
        }
        
        // Se regresa la matriz transpuesta
        return transposeMatrix;
    }
    
    /// <summary>
    /// Calcula el promedio a lo largo de una dimensión específica.
    /// </summary>
    /// <param name="matrix">Matriz bidimensional sobre la cual calcular el promedio.</param>
    /// <param name="dimension">La dimensión a lo largo de la cual calcular el promedio.</param>
    /// <returns>Arreglo con los promedios calculados.</returns>
    public static double[] AverageAlongDimension(double[,] matrix, int dimension)
    {
        var size = new[] { matrix.GetLength(0), matrix.GetLength(1) };
        
        // Se declara la nueva matriz (de una dimensión)
        var newMatrix = new double[size[1 - dimension]];
        
        // Dependiendo de la dimensión escogida
        if (dimension == 1)
        {
            // Por cada elemento
            for (var i = 0; i < size[0]; i++)
            {
                // Se suman las columnas
                for (var j = 0; j < size[1]; j++)
                {
                    newMatrix[i] += matrix[i, j];
                }
                
                // Y se promedian
                newMatrix[i] /= size[1];
            }
        }
        else
        {
            // Por cada elemento
            for (var i = 0; i < size[1]; i++)
            {
                // Se suman las filas
                for (var j = 0; j < size[0]; j++)
                {
                    newMatrix[i] += matrix[j, i];
                }
                
                // Y se promedian
                newMatrix[i] /= size[0];
            }
        }
        
        // Se regresa la matriz reducida (Arreglo)
        return newMatrix;
    }
}