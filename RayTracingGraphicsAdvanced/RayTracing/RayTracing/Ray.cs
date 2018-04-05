using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RayTracing
{
    public class Ray
    {
        public P3 origin, direction;
        public Ray(P3 origin, P3 direction)
        {
            this.origin = origin;
            this.direction = direction.norm();
        }
        public P3 travel(float t)
        {
            return origin.add(direction.scale(t));
        }
    }
}
