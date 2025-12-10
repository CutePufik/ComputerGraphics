namespace cornish_room
{
    public static class ColorExtensions
    {
        public static Color Multiply(this Color color, float factor)
        {
            int r = (int)(color.R * factor);
            int g = (int)(color.G * factor);
            int b = (int)(color.B * factor);

            Clamp(ref r, ref g, ref b);

            return Color.FromArgb(r, g, b);
        }

        public static Color Add(this Color color, Color other)
        {
            int r = color.R + other.R;
            int g = color.G + other.G;
            int b = color.B + other.B;

            Clamp(ref r, ref g, ref b);

            return Color.FromArgb(r, g, b);
        }

        private static void Clamp(ref int r, ref int g, ref int b)
        {
            r = Math.Min(255, Math.Max(0, r));
            g = Math.Min(255, Math.Max(0, g));
            b = Math.Min(255, Math.Max(0, b));
        }
    }
}
