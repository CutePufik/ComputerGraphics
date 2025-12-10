namespace cornish_room;

public class Light
{
    public Point3D Position { get; set; }
    public float Intensity { get; set; }

    public Light(Point3D position, float intensity)
    {
        Position = position;
        Intensity = intensity;
    }
}