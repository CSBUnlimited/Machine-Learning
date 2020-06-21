using Microsoft.ML.Data;

namespace LinearRegression
{
    public  class SalaryData
    {
        [LoadColumn(0)]
        public float YearsExperience { get; set; }

        [LoadColumn(1), ColumnName("Label")]
        public float Salary { get; set; }
    }
}