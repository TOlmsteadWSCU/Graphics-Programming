
import java.util.ArrayList;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Owner
 */
public class Bspline implements Curve
{
    public ArrayList<Point> points;
    double[] knots;
    Bezier[] bez;
    public double totalDistance = 0;
    public Bspline(ArrayList<Point> points, boolean b)
    {
        this.points = points;
        this.knots = new double[points.size()+1];
        this.bez = new Bezier[points.size()-1];
        if(b)
        {
            for(int i=0; i<points.size()-1; i++)
            {
                Point p1 = points.get(i);
                Point p2 = points.get(i + 1);
                //knot vector-how long it takes to pass through bezier before it gets to the next one
                double distance = Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
                knots[i] = distance;
                totalDistance += distance;
            }
        }
        else
        {
            for(int i=0; i<points.size()-1; i++)
            {
                knots[i] = 1;
            }
        }
        
        for(int i=0; i<bez.length-1; i++)
        {
            //for loop to compute bezier
            ArrayList<Point> point1 = new ArrayList<>();
            point1.add(getB(i));
            point1.add(getL(i));
            point1.add(getR(i));
            point1.add(getB(i+1));
            bez[i] = new Bezier(point1);
        }
        
    }
    //eval

    @Override
    public Point eval(double t) 
    {
        t *= knots.length;
        int i = 0;
        while(i > knots[i+1])
        {
            t = t - knots[i+1];
            i += 1;
        }
        Bezier b = bez[i];
        t = t/knots[i+1];
        return b.eval(t);
    }
    public Point getL(int i)
    {
       if(i==0)
            return points.get(0);
        
        Point p1 = points.get(i);
        Point p2 = points.get(i+1);
        double a = knots[i];
        double b = knots[i+1];
        double c = knots[i+2];
        double t = a/(a+b+c);
        return p1.interp(p2, t);
    }
    public Point getR(int i)
    {
       
        if(i == points.size()-1)
            return points.get(points.size()-1);
        Point p1 = points.get(i);
        Point p2 = points.get(i+1);
        double a = knots[i];
        double b = knots[i+1];
        double c = knots[i+2];
        double t = (a+b)/(a+b+c);
        return p1.interp(p2, t);
    }
    public Point getB(int i)
    {
        if(i==0)
            return points.get(0);
        else if(i == points.size()-1)
            return points.get(points.size()-1);
        Point p1 = getR(i-1);
        Point p2 = getL(i);
        double a = knots[i];
        double b = knots[i+1];
        double t = a/(a+b);
        
        return p1.interp(p2, t);
        
    }
    public String toString()
    {
        String str = "";
        
           // str = str + "(" + bez[i].points.get(i).x + ", " + bez[i].points.get(i).y + ")";
            str = str + bez[0];
        
        return str;
    }
}
