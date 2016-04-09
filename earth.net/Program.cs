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
            rnet.RNetInterop r = new rnet.RNetInterop();
            var dtOzone = r.ReadData();

            var dt = TestingEnvironment.CreateCarsTable();
            Earth e = new Earth(dtOzone);
            var result =  e.Predict("V4");

        }
    }
}
