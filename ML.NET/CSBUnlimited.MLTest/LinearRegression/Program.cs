using Microsoft.ML;
using System;

namespace LinearRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            // Load Data
            var trainData = context.Data.LoadFromTextFile<SalaryData>("SalaryData.csv", hasHeader: true, separatorChar: ',');

            var testTrainSplit = context.Data.TrainTestSplit(trainData, 0.2);

            // Build Model
            var pipeline = context.Transforms.Concatenate("Features", "YearsExperience")
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(testTrainSplit.TrainSet);

            // Evaluate model
            var predictions = model.Transform(testTrainSplit.TestSet);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine("R^2 - {0}", metrics.RSquared);

            // Predict
            var newData = new SalaryData()
            {
                YearsExperience = 1.1F
            };

            var predictionFunc = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model);

            var prediction = predictionFunc.Predict(newData);
            Console.WriteLine("Prediction - {0}", prediction.PredictedSalary);

            Console.ReadLine();
        }
    }
}
