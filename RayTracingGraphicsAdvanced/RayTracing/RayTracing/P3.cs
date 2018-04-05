using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RayTracing
{
    public class P3
    {
        public double x, y, z;
        public P3(double x, double y, double z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        public P3 add(P3 p)
        {
            return new P3(x + p.x, y + p.y, z + p.z);
        }
        public P3 sub(P3 p)
        {
            return new P3(x - p.x, y - p.y, z - p.z);
        }
        public double abs()
        {
            return Math.Sqrt(x * x + y * y + z * z);
        }
        public P3 scale(double d)
        {
            return new P3(d * x, d * y, d * z);
        }
        public P3 norm()
        {
            double d = abs();
            if (d == 0)
            {
                return new P3(0, 0, 0);
            }
            return scale(1 / d);
        }
        public String toString()
        {
            return "(" + x + "," + y + "," + z + ")";
        }
        public double dot(P3 p)
        {
            return x * p.x + y * p.y + z * p.z;
        }
        public P3 cross(P3 p)
        {
            return new P3(y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x);
        }
        
    }
}
