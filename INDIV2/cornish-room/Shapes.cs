namespace cornish_room;

public abstract class Shape
{
    public Color Color { get; set; }
    public bool Specularity { get; set; }
    public bool Transparency { get; set; }
    public abstract (bool collide, float dist, Point3D point, Point3D normal, Shape shape) Collision(Ray ray);
}

public class Sphere : Shape
{
    public Point3D Position { get; set; }
    public float Radius { get; set; }

    public Sphere(Point3D position, float radius, Color color, bool specularity, bool transparency)
    {
        Position = position;
        Radius = radius;
        Color = color;
        Specularity = specularity;
        Transparency = transparency;
    }

    public override (bool collide, float dist, Point3D point, Point3D normal, Shape shape) Collision(Ray ray)
    {
        Point3D L = Position - ray.Origin;
        float tca = L.DotProduct(ray.Direction);  // Проекция на луч
        float d2 = L.DotProduct(L) - tca * tca;   // Расстояние от луча до центра (в квадрате)

        float r2 = Radius * Radius;

        // Луч проходит мимо сферы
        if (d2 > r2)
            return (false, float.PositiveInfinity, null, null, this);

        float thc = (float)Math.Sqrt(r2 - d2);  // Половина длины хорды

        float t0 = tca - thc;  // Ближнее пересечение
        float t1 = tca + thc;  // Дальнее пересечение

        // Выбираем правильный корень
        float t;
        if (t0 > 0.001f)
            t = t0;           // Ближнее пересечение впереди
        else if (t1 > 0.001f)
            t = t1;           // Луч внутри сферы, берём выход
        else
            return (false, float.PositiveInfinity, null, null, this);  // Оба позади

        Point3D point = ray.Origin + ray.Direction * t;
        Point3D normal = (point - Position).Normalize();

        // Если луч внутри сферы, нормаль направлена внутрь
        if (t0 < 0)
            normal = normal * -1;

        return (true, t, point, normal, this);
    }
}

public class Triangle : Shape
{
    public List<Point3D> Points { get; set; }

    public Triangle(List<Point3D> points, Color color, bool specularity, bool transparency)
    {
        Points = points;
        Color = color;
        Specularity = specularity;
        Transparency = transparency;
    }

    public override (bool collide, float dist, Point3D point, Point3D normal, Shape shape) Collision(Ray ray)
    {
        // Найти пересечение с плоскостью треугольника
        Point3D normal = (Points[1] - Points[0]).CrossProduct(Points[2] - Points[0]).Normalize();
        float faceOffset = normal.DotProduct(Points[0]);
        float distToSurface = (faceOffset - normal.DotProduct(ray.Origin)) / normal.DotProduct(ray.Direction);
        Point3D point = ray.Origin + ray.Direction * distToSurface;

        bool c1 = normal.DotProduct((Points[1] - Points[0]).CrossProduct(point - Points[0])) >= 0;
        bool c2 = normal.DotProduct((Points[2] - Points[1]).CrossProduct(point - Points[1])) >= 0;
        bool c3 = normal.DotProduct((Points[0] - Points[2]).CrossProduct(point - Points[2])) >= 0;

        //Проверить, внутри ли треугольника точка
        bool collide = c1 && c2 && c3;

        if (!collide || distToSurface <= 0)
        {
            distToSurface = float.PositiveInfinity;
        }
        // было ли пересечение, расстояние до точки пересечения, координаты точки пересечения, нормаль поверхности в этой точке, ссылка на сам объект
        return (collide, distToSurface, point, normal, this);
    }
}

public class Face : Shape
{
    public List<Triangle> Triangles { get; set; }

    public Face(List<Point3D> points, Color color, bool specularity, bool transparency)
    {
        Color = color;
        Specularity = specularity;
        Transparency = transparency;
        Triangulate(points);
    }

    private void Triangulate(List<Point3D> points)
    {
        Triangles = new List<Triangle>
        {
            new Triangle(new List<Point3D> { points[0], points[1], points[2] }, Color, Specularity, Transparency),
            new Triangle(new List<Point3D> { points[2], points[3], points[0] }, Color, Specularity, Transparency)
        };
    }

    public override (bool collide, float dist, Point3D point, Point3D normal, Shape shape) Collision(Ray ray)
    {
        var collision = Triangles[0].Collision(ray);
        for (int i = 1; i < Triangles.Count; ++i)
        {
            var nextCollision = Triangles[i].Collision(ray);
            if (nextCollision.dist < collision.dist)
            {
                collision = nextCollision;
            }
        }
        collision.shape = this;
        return collision;
    }
}

public class Cube : Shape
{
    public Point3D Position { get; set; }
    public float ScaleX { get; set; }
    public float ScaleY { get; set; }
    public float ScaleZ { get; set; }
    public List<Face> Faces { get; set; }

    public Cube(Point3D position, float scaleX, float scaleY, float scaleZ, Color color, bool specularity, bool transparency)
    {
        Position = position;
        ScaleX = scaleX;
        ScaleY = scaleY;
        ScaleZ = scaleZ;
        Color = color;
        Specularity = specularity;
        Transparency = transparency;
        SetFaces();
    }

    private void SetFaces()
    {
        float x = ScaleX / 2;
        float y = ScaleY / 2;
        float z = ScaleZ / 2;

        List<Point3D> points = new List<Point3D>
        {
            new Point3D(-x, +y, -z) + Position, 
            new Point3D(-x, +y, +z) + Position,
            new Point3D(+x, +y, +z) + Position, 
            new Point3D(+x, +y, -z) + Position, 
            new Point3D(-x, -y, -z) + Position, 
            new Point3D(-x, -y, +z) + Position, 
            new Point3D(+x, -y, +z) + Position,
            new Point3D(+x, -y, -z) + Position  
        };

        Faces = new List<Face>
        {
            new Face(new List<Point3D> { points[0], points[1], points[2], points[3] }, Color, Specularity, Transparency),
            new Face(new List<Point3D> { points[4], points[5], points[6], points[7] }, Color, Specularity, Transparency),
            new Face(new List<Point3D> { points[0], points[3], points[7], points[4] }, Color, Specularity, Transparency),
            new Face(new List<Point3D> { points[1], points[5], points[6], points[2] }, Color, Specularity, Transparency),
            new Face(new List<Point3D> { points[0], points[1], points[5], points[4] }, Color, Specularity, Transparency),
            new Face(new List<Point3D> { points[3], points[2], points[6], points[7] }, Color, Specularity, Transparency)
        };
    }
    
    public override (bool collide, float dist, Point3D point, Point3D normal, Shape shape) Collision(Ray ray)
    {
        var collision = Faces[0].Collision(ray);
        for (int i = 1; i < Faces.Count; ++i)
        {
            var nextCollision = Faces[i].Collision(ray);
            if (nextCollision.dist < collision.dist)
            {
                collision = nextCollision;
            }
        }
        collision.shape = this;
        return collision;
    }
}
