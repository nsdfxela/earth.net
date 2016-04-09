using RDotNet;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace earth.net.rnet
{
    internal class RNetInterop
    {
        public DataTable ReadData()
        {
            REngine engine = REngine.GetInstance();
            
            var O = engine.Evaluate(@"library(mlbench)
                                    data(Ozone)
                                    t <- na.omit(Ozone)").AsDataFrame();
            DataTable dt = new DataTable();
            for (int i = 0; i < O.ColumnCount; i++)
            {
                dt.Columns.Add(O.ColumnNames[i], typeof(double));
            }

            for (int i = 0; i < O.RowCount; i++)
            {
                List<object> row = new List<object>();
                for (int j = 0; j < O.ColumnCount; j++)
                    row.Add(O[i,j]);
                dt.Rows.Add(row.ToArray());
            }
            return dt;
        }
    }
}
