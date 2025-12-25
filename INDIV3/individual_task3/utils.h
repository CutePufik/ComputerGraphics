#ifndef UTILS_H
#define UTILS_H

#include <GL/glew.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ======================= МАТЕМАТИКА =======================
struct Vec2 {
    float x = 0, y = 0;
};

struct Vec3
{
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float X, float Y, float Z) :x(X), y(Y), z(Z) {}
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline Vec3 operator*(const Vec3& a, float s) {
    return { a.x * s, a.y * s, a.z * s };
}

inline float Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 Cross(const Vec3& a, const Vec3& b)
{
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline float Length(const Vec3& v) {
    return std::sqrt(Dot(v, v));
}

inline Vec3 Normalize(const Vec3& v)
{
    float len = Length(v);
    if (len <= 1e-6f) return v;
    return v * (1.0f / len);
}

// column-major Mat4
struct Mat4
{
    float m[16] = { 0 };

    static Mat4 Identity()
    {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    static Mat4 Translation(float x, float y, float z)
    {
        Mat4 r = Identity();
        r.m[12] = x; r.m[13] = y; r.m[14] = z;
        return r;
    }

    static Mat4 Scale(float x, float y, float z)
    {
        Mat4 r;
        r.m[0] = x; r.m[5] = y; r.m[10] = z; r.m[15] = 1.0f;
        return r;
    }

    static Mat4 RotationY(float a)
    {
        Mat4 r = Identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[0] = c;  r.m[2] = s;
        r.m[8] = -s; r.m[10] = c;
        return r;
    }

    static Mat4 RotationX(float a)
    {
        Mat4 r = Identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[5] = c;  r.m[6] = s;
        r.m[9] = -s; r.m[10] = c;
        return r;
    }

    static Mat4 Perspective(float fovyRad, float aspect, float zNear, float zFar)
    {
        Mat4 r;
        float t = std::tan(fovyRad / 2.0f);
        r.m[0] = 1.0f / (aspect * t);
        r.m[5] = 1.0f / t;
        r.m[10] = -(zFar + zNear) / (zFar - zNear);
        r.m[11] = -1.0f;
        r.m[14] = -(2.0f * zFar * zNear) / (zFar - zNear);
        return r;
    }

    static Mat4 LookAt(const Vec3& eye, const Vec3& center, const Vec3& up)
    {
        Vec3 f = Normalize(center - eye);
        Vec3 s = Normalize(Cross(f, up));
        Vec3 u = Cross(s, f);

        Mat4 r = Identity();
        r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
        r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
        r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;

        r.m[12] = -Dot(s, eye);
        r.m[13] = -Dot(u, eye);
        r.m[14] = Dot(f, eye);
        return r;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b)
{
    Mat4 r;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
        {
            r.m[col * 4 + row] =
                a.m[0 * 4 + row] * b.m[col * 4 + 0] +
                a.m[1 * 4 + row] * b.m[col * 4 + 1] +
                a.m[2 * 4 + row] * b.m[col * 4 + 2] +
                a.m[3 * 4 + row] * b.m[col * 4 + 3];
        }
    return r;
}

// ======================= ЛОГИ ШЕЙДЕРОВ =======================
inline void ShaderLog(GLuint shader)
{
    GLint infologLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLen);
    if (infologLen > 1)
    {
        std::vector<char> infoLog(infologLen);
        GLsizei charsWritten = 0;
        glGetShaderInfoLog(shader, infologLen, &charsWritten, infoLog.data());
        std::cout << "Shader log:\n" << infoLog.data() << std::endl;
    }
}

inline void ProgramLog(GLuint prog)
{
    GLint infologLen = 0;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infologLen);
    if (infologLen > 1)
    {
        std::vector<char> infoLog(infologLen);
        GLsizei charsWritten = 0;
        glGetProgramInfoLog(prog, infologLen, &charsWritten, infoLog.data());
        std::cout << "Program log:\n" << infoLog.data() << std::endl;
    }
}

inline GLuint CompileShader(GLenum type, const char* src)
{
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    ShaderLog(sh);
    return sh;
}

inline GLuint LinkProgram(GLuint vert, GLuint frag)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    GLint success = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) ProgramLog(prog);
    return prog;
}

// ======================= ТЕКСТУРЫ =======================
#include <SFML/Graphics/Image.hpp>

inline GLuint LoadTextureFromFile(const std::string& filename, bool srgb = false)
{
    sf::Image img;
    if (!img.loadFromFile(filename))
    {
        std::cout << "Failed to load texture: " << filename << std::endl;
        return 0;
    }
    img.flipVertically();

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    GLint internalFmt = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;

    glTexImage2D(GL_TEXTURE_2D, 0, internalFmt,
        (GLsizei)img.getSize().x, (GLsizei)img.getSize().y,
        0, GL_RGBA, GL_UNSIGNED_BYTE, img.getPixelsPtr());

    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

// ======================= OBJ (pos+uv+normal) + тангенты =======================
// Выходной формат вершины: pos(3) normal(3) uv(2) tangent(3) = 11 float
struct VertexPNUt
{
    Vec3 p;
    Vec3 n;
    Vec2 uv;
    Vec3 t; // tangent
};

inline float frand(float a, float b)
{
    return a + (b - a) * (rand() / (float)RAND_MAX);
}

inline bool LoadOBJ_PNU(const std::string& filename, std::vector<VertexPNUt>& out)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "Failed to open OBJ: " << filename << std::endl;
        return false;
    }

    std::vector<Vec3> positions;
    std::vector<Vec2> texcoords;
    std::vector<Vec3> normals;

    struct Idx { int vi = 0, ti = 0, ni = 0; };

    auto parseIndex = [&](const std::string& s)->Idx
        {
            Idx r;
            size_t a = s.find('/');
            if (a == std::string::npos)
            {
                r.vi = std::stoi(s);
                return r;
            }
            std::string vStr = s.substr(0, a);
            if (!vStr.empty()) r.vi = std::stoi(vStr);

            size_t b = s.find('/', a + 1);
            std::string vtStr, vnStr;
            if (b == std::string::npos)
            {
                vtStr = s.substr(a + 1);
            }
            else
            {
                vtStr = s.substr(a + 1, b - a - 1);
                vnStr = s.substr(b + 1);
            }
            if (!vtStr.empty()) r.ti = std::stoi(vtStr);
            if (!vnStr.empty()) r.ni = std::stoi(vnStr);
            return r;
        };

    std::string line;
    std::vector<Idx> face;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string pref; iss >> pref;
        if (pref == "v")
        {
            float x, y, z; iss >> x >> y >> z;
            positions.emplace_back(x, y, z);
        }
        else if (pref == "vt")
        {
            float u, v; iss >> u >> v;
            texcoords.push_back({ u,v });
        }
        else if (pref == "vn")
        {
            float x, y, z; iss >> x >> y >> z;
            normals.emplace_back(x, y, z);
        }
        else if (pref == "f")
        {
            face.clear();
            std::string tok;
            while (iss >> tok) face.push_back(parseIndex(tok));
            if (face.size() < 3) continue;

            // fan triangulation
            for (size_t i = 1; i + 1 < face.size(); ++i)
            {
                Idx i0 = face[0];
                Idx i1 = face[i];
                Idx i2 = face[i + 1];

                auto getP = [&](int vi)->Vec3 { return positions[std::max(1, vi) - 1]; };
                auto getUV = [&](int ti)->Vec2 {
                    if (ti <= 0 || ti > (int)texcoords.size()) return { 0,0 };
                    return texcoords[ti - 1];
                    };
                auto getN = [&](int ni)->Vec3 {
                    if (ni <= 0 || ni > (int)normals.size()) return { 0,1,0 };
                    return normals[ni - 1];
                    };

                VertexPNUt v0, v1, v2;
                v0.p = getP(i0.vi); v1.p = getP(i1.vi); v2.p = getP(i2.vi);
                v0.uv = getUV(i0.ti); v1.uv = getUV(i1.ti); v2.uv = getUV(i2.ti);

                // нормали: если нет vn — посчитаем face normal
                Vec3 faceN = Normalize(Cross(v1.p - v0.p, v2.p - v0.p));
                v0.n = (i0.ni ? getN(i0.ni) : faceN);
                v1.n = (i1.ni ? getN(i1.ni) : faceN);
                v2.n = (i2.ni ? getN(i2.ni) : faceN);

                // tangent по треугольнику
                Vec3 e1 = v1.p - v0.p;
                Vec3 e2 = v2.p - v0.p;
                float du1 = v1.uv.x - v0.uv.x;
                float dv1 = v1.uv.y - v0.uv.y;
                float du2 = v2.uv.x - v0.uv.x;
                float dv2 = v2.uv.y - v0.uv.y;

                float f = (du1 * dv2 - du2 * dv1);
                Vec3 tangent = { 1,0,0 };
                if (std::fabs(f) > 1e-8f)
                {
                    float inv = 1.0f / f;
                    tangent = Normalize((e1 * dv2 - e2 * dv1) * inv);
                }

                v0.t = tangent; v1.t = tangent; v2.t = tangent;

                out.push_back(v0);
                out.push_back(v1);
                out.push_back(v2);
            }
        }
    }

    if (out.empty())
    {
        std::cout << "OBJ has no triangles: " << filename << std::endl;
        return false;
    }

    std::cout << "OBJ loaded: " << filename << ", vertices: " << out.size() << std::endl;
    return true;
}

// ======================= MESH =======================
struct Mesh
{
    GLuint VAO = 0;
    GLuint VBO = 0;
    GLsizei vertexCount = 0;
    float minY = 0.0f;
};

inline Mesh CreateMesh_PNUt(const std::vector<VertexPNUt>& verts)
{
    Mesh m;
    m.vertexCount = (GLsizei)verts.size();

    // minY
    float mn = 1e9f;
    for (const auto& v : verts) mn = std::min(mn, v.p.y);
    m.minY = mn;

    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);

    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(VertexPNUt), verts.data(), GL_STATIC_DRAW);

    // location 0: pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPNUt), (void*)offsetof(VertexPNUt, p));
    glEnableVertexAttribArray(0);
    // location 1: normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPNUt), (void*)offsetof(VertexPNUt, n));
    glEnableVertexAttribArray(1);
    // location 2: uv
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexPNUt), (void*)offsetof(VertexPNUt, uv));
    glEnableVertexAttribArray(2);
    // location 3: uv
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPNUt), (void*)offsetof(VertexPNUt, t));
    glEnableVertexAttribArray(3);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return m;
}

inline Mesh CreateGround(float size, float uvScale)
{
    std::vector<VertexPNUt> v;
    v.resize(6);

    Vec3 n = { 0,1,0 };
    Vec3 t = { 1,0,0 };

    float s = size;
    // два треугольника
    v[0] = { {-s,0,-s}, n, {0,0}, t };
    v[1] = { { s,0, s},  n, {uvScale,uvScale}, t };
    v[2] = { { s,0,-s}, n, {uvScale,0}, t };

    v[3] = { {-s,0,-s}, n, {0,0}, t };
    v[4] = { {-s,0, s}, n, {0,uvScale}, t };
    v[5] = { { s,0, s},  n, {uvScale,uvScale}, t };

    return CreateMesh_PNUt(v);
}

// ======================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =======================

/// Преобразование градусов в радианы
inline float deg2rad(float d) {
    return d * (float)M_PI / 180.0f;
}


/// Проверка столкновения сфер: расстояние <= (радиус1 + радиус2)
inline bool SphereHit(const Vec3& a, float ra, const Vec3& b, float rb)
{
    float d = Length(a - b);
    return d <= (ra + rb);
}

#endif 