using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace earth.net
{
    class Program
    {
        static void Main(string[] args)
        {
            var dt = TestingEnvironment.CreateCarsTable();
            Earth e = new Earth(dt);
            var result =  e.Predict("mpg");

        }
    }
}
