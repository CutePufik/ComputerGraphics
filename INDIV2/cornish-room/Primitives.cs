namespace cornish_room
{
    public class Point3D    
    {
        public float X, Y, Z, W;

        public Point3D(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
            W = 1;
        }
        public Point3D Clone()
        {
            return new Point3D(this.X, this.Y, this.Z) { W = this.W };
        }
        public static Point3D operator /(Point3D p1, float scalar)
        {
            if (scalar == 0.0f)
                throw new DivideByZeroException();
            return new Point3D(p1.X / scalar, p1.Y / scalar, p1.Z / scalar);
        }
        public static Point3D operator *(Point3D p1, float scalar)
        {
            return new Point3D(p1.X * scalar, p1.Y * scalar, p1.Z * scalar);
        }
        public static Point3D operator -(Point3D p1, Point3D p2)
        {
            return new Point3D(p1.X - p2.X, p1.Y - p2.Y, p1.Z - p2.Z);
        }
        public static Point3D operator +(Point3D p1, Point3D p2)
        {
            return new Point3D(p1.X + p2.X, p1.Y + p2.Y, p1.Z + p2.Z);
        }
        public Point3D CrossProduct(Point3D other)
        {
            return new Point3D(
                this.Y * other.Z - this.Z * other.Y,
                this.Z * other.X - this.X * other.Z,
                this.X * other.Y - this.Y * other.X
            );
        }
        public float DotProduct(Point3D other)
        {
            return this.X * other.X + this.Y * other.Y + this.Z * other.Z;
        }
        public float Length()
        {
            return (float)Math.Sqrt(X * X + Y * Y + Z * Z);
        }
        public Point3D Normalize()
        {
            float length = Length();
            if (length > 0)
            {
                X /= length;
                Y /= length;
                Z /= length;
            }
            return this;
        }
    }
}