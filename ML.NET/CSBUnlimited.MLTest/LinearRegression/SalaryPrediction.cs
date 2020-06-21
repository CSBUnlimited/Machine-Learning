using Microsoft.ML.Data;

namespace LinearRegression
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary { get; set; }
    }
}