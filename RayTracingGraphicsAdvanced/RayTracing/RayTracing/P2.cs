using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RayTracing
{
    public class P2
    {
        public double x, y;
        public P2(double x, double y)
        {
            this.x = x;
            this.y = y;
        }
        public String toString()
        {
            return "(" + x + "," + y + ")";
        }
    }
}
