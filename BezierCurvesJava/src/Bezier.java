
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
public class Bezier implements Curve 
{
    public ArrayList<Point> points;
    public Bezier(ArrayList<Point> points)
    {
        this.points = points;
    }
    
    public static Point eval1(ArrayList<Point> p, double t)
    {
        if(p.size()==1){
            return p.get(0);
        }
        return eval1(pts(p, t), t);
    }
    @Override
    public Point eval(double t)
    {
        return eval1(points, t);
    }
    public static ArrayList<Point> pts(ArrayList<Point> points, double t)
    {
        ArrayList<Point> result = new ArrayList<>();
        for(int i=0; i<points.size()-1; i++)
        {
            Point p1 = points.get(i);
            Point p2 = points.get(i + 1);
            result.add(p1.interp(p2, t));
        }
        return result;
        
    }
    public String toString()
    {
        String str = "";
        for(int i=0; i<points.size(); i++)
        {
            str = str + "(" + points.get(i).x + ", " + points.get(i).y + ")";
        }
        return str;
    }
}
