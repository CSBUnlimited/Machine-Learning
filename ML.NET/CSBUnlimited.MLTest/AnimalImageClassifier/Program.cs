﻿using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace AnimalImageClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            // Define Paths
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            var data = Path.Combine(projectDirectory, "data");

            // Initialize instances
            MLContext mlContext = new MLContext();

            // Load data
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: data, useFolderNameAsLabel: true);

            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

            // Build Model
            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelAsKey")
                .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: data, inputColumnName: "ImagePath"));

            IDataView preProcessedData = preprocessingPipeline
                .Fit(shuffledData)
                .Transform(shuffledData);

            // Train test data
            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.25);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;

            // Train Model
            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true,
                WorkspacePath = workspaceRelativePath
            };

            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(trainSet);

            // Save model
            mlContext.Model.Save(trainedModel, imageData.Schema, "animal-classifier-model.zip");

            ClassifyImages(mlContext, testSet, trainedModel);

            Console.ReadLine();
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

            string[] extentions = new string[] { ".jpg", ".jpeg", ".png" };

            foreach (var file in files)
            {
                if (!extentions.Any((extension) => Path.GetExtension(file) == extension))
                {
                    continue;
                }

                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }

        private static void OutputPrediction(ModelOutput prediction)
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        }

        public static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
            ModelOutput prediction = predictionEngine.Predict(image);

            Console.WriteLine("Classifying single image");
            OutputPrediction(prediction);
        }

        public static void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            IDataView predictionData = trainedModel.Transform(data);
            IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);

            Console.WriteLine("Classifying multiple images");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }
        }
    }
}