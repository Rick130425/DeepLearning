using DeepLearning.ML;
using DeepLearning.ML.LossFunctions;
using DeepLearning.ML.Nodes.HiddenLayers;
using DeepLearning.ML.Optimizers;
using static DeepLearning.ML.MLOperations;

// Se declara la red neuronal que tomará 3 datos de entrada y tendrá como función de perdida la función MSE
var neuralNetwork = new ML(3, new MeanSquaredError());

// Después se aplica una capa ReLU de 10 nodos con optimizador SGD
neuralNetwork.AddLayer(new ReLU(10, new SGD()));
// Se aplica una capa ReLU de 10 nodos con optimizador SGD
neuralNetwork.AddLayer(new ReLU(5, new SGD()));
// Se aplica otra capa ReLU de un nodo con optimizador SGD
neuralNetwork.AddLayer(new ReLU(1, new SGD()));

// Se leen los datos de demostración de su archivo csv correspondiente
var data = ReadNumericDataFromCsv("./Datasets/demo.csv", new[] { 0, 2 }, new[] { 3, 3 }, 
    hasHeaders: true);

// Se imprime la consola inicial
Console.Write("""
                  ¡Bienvenido a la demo demostrativa!
                  -----------------------------------------------------------
                  Como demostración de su funcionamiento se entrenará una red 
                  neuronal con la siguiente estructura:
                  [
                  ReLU de 10 nodos con Optimizador SGD
                  ReLU de 5 nodos con Optimizador SGD
                  ReLU de 1 nodo con Optimizador SGD
                  ]
                  
                  Se entrenará con un dataset de prueba de 256 datos 
                  (3 entradas y 1 salida)
                  -----------------------------------------------------------
                  Primero se imprime su pérdida antes del entrenamiento: 
                  """);
// Se imprime el promedio de pérdida en el dataset
Console.WriteLine(neuralNetwork.AverageLoss(data));

Console.WriteLine("""
                  
                  Después de esto se entrena a 1000 épocas con un batchSize de 
                  32 datos:
                  """);
// Se realizan 10 entrenamientos de 100 epochs cada uno
for (var i = 0; i < 10; i++)
{
    neuralNetwork.Train(data, 32, 100);
    // Y se imprime el promedio de pérdida en cada uno
    Console.WriteLine($"Pérdida ({i * 10 + 10}): " + neuralNetwork.AverageLoss(data));
}

Console.WriteLine("""
                  
                  Por último, se imprimen 10 casos para ver su generación de datos:
                  """);

// Se realizan 10 casos de prueba
for (var i = 0; i < 10; i++)
{
    Console.WriteLine($"\nCaso {i + 1}:");
    // Donde se imprime el valor calculado
    Console.WriteLine($"Valor calculado: \t{neuralNetwork.ForwardPropagation(data[i][0])[0]}");
    // Y el valor real
    Console.WriteLine($"Valor real: \t\t{data[i][1][0]}");
}

// Por último se una explicación breve y se menciona la finalización de la demo demostrativa
Console.WriteLine("""
                  
                  Como se puede apreciar, los valores de la red neuronal
                  son parecidos a los valores reales, pero no exactos.
                  
                  Una de las particularidades es que entre más se 
                  entrene una red neuronal, mayor será su precisión en 
                  los resultados (aunque existen excepciones), por lo que
                  al entrenar la red neuronal, esta se acercó a los valores
                  reales, pero al no entrenar demasiado (para evitar tiempos
                  de espera altos), no se consiguieron resultados exactos.
                  -----------------------------------------------------------
                  
                  Con esto concluye la demo demostrativa.
                  
                  ¡Esperamos a que te animes a hacer tus 
                  propias redes neuronales en la versión completa!
                  """);