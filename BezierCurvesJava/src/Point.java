/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Owner
 */
public class Point {
    public double x, y;
    public Point(double x, double y)
    {
        this.x = x;
        this.y = y;
    }
    public Point interp(Point p, double t)
    {
        return new Point(((1-t)*this.x + t*p.x), ((1-t)*this.y + t*p.y));
    }
}
