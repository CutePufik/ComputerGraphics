#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics/Image.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void ShaderLog(GLuint shader)
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

void ProgramLog(GLuint prog)
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

void CheckOpenGLError(const char* where)
{
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR)
    {
        std::cout << "OpenGL error at " << where << ": 0x"
            << std::hex << err << std::dec << std::endl;
    }
}

GLuint CompileShader(GLenum type, const char* src)
{
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    ShaderLog(sh);
    return sh;
}

GLuint LinkProgram(GLuint vert, GLuint frag)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    GLint success = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success)
    {
        ProgramLog(prog);
    }
    return prog;
}

// --------------------- ШЕЙДЕРЫ ---------------------

// Вершинный шейдер для 3D объектов (тетраэдр, кубы)
const char* vertex3DShaderSrc = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;
    layout (location = 2) in vec2 aTexCoord;

    out vec3 vColor;
    out vec2 vTexCoord;

    uniform vec3 uOffset; // смещение модели

    void main()
    {
        vColor = aColor;
        vTexCoord = aTexCoord;

        // позиция в "мире"
        vec3 pos = aPos + uOffset;

        gl_Position = vec4(pos, 1.0);
    }
)";



// Фрагментный шейдер: градиент по цветам вершин (для тетраэдра)
const char* fragTetraSrc = R"(
    #version 330 core
    in vec3 vColor;
    out vec4 FragColor;
    void main()
    {
        FragColor = vec4(vColor, 1.0);
    }
)";

// Фрагментный шейдер: одна текстура + влияние цвета вершин
const char* fragCubeColorTexSrc = R"(
    #version 330 core
    in vec3 vColor;
    in vec2 vTexCoord;
    out vec4 FragColor;

    uniform sampler2D uTexture;
    uniform float uColorFactor;

    void main()
    {
        vec4 texColor = texture(uTexture, vTexCoord);
        vec4 vertColor = vec4(vColor, 1.0);
        FragColor = mix(texColor, vertColor, uColorFactor);
    }
)";

// Фрагментный шейдер: две текстуры, смешивание
const char* fragCubeTwoTexSrc = R"(
    #version 330 core
    in vec2 vTexCoord;
    in vec3 vColor;
    out vec4 FragColor;

    uniform sampler2D uTexture1;
    uniform sampler2D uTexture2;
    uniform float uMixFactor;

    void main()
    {
        vec4 t1 = texture(uTexture1, vTexCoord);
        vec4 t2 = texture(uTexture2, vTexCoord);
        FragColor = mix(t1, t2, uMixFactor);
    }
)";

// Вершинный шейдер для круга (2D)
const char* vertexCircleShaderSrc = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;

    out vec2 vPos;

    uniform vec2 uScale;
    uniform vec2 uOffset;

    void main()
    {
        vPos = aPos;
        vec2 scaled = aPos * uScale + uOffset;
        gl_Position = vec4(scaled, 0.0, 1.0);
    }
)";

// Фрагментный шейдер для градиентного круга (HSV Hue по углу, центр белый)
const char* fragCircleSrc = R"(
    #version 330 core
    in vec2 vPos;
    out vec4 FragColor;

    // Конвертация HSV -> RGB, c.x = H [0..1], c.y = S, c.z = V
    vec3 hsv2rgb(vec3 c)
    {
        vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    void main()
    {
        float r = length(vPos); // расстояние до центра
        float maxRadius = 0.5;  // радиус, на котором цвет полностью насыщён

        // hue по углу (atan)
        float angle = atan(vPos.y, vPos.x); // [-pi, pi]
        float hue = angle / (2.0 * 3.14159265358979323846) + 0.5; // [0..1]

        vec3 rainbow = hsv2rgb(vec3(hue, 1.0, 1.0));

        // центр белый, к краю переходим к rainbow
        float t = clamp(r / maxRadius, 0.0, 1.0);
        vec3 color = mix(vec3(1.0, 1.0, 1.0), rainbow, t);

        // за пределами круга — просто делаем прозрачное/чёрное
        if (r > maxRadius)
            discard;

        FragColor = vec4(color, 1.0);
    }
)";

// --------------------- МЕШИ / VBO ---------------------

struct Mesh3D {
    GLuint VAO = 0;
    GLuint VBO = 0;
    GLenum primitive = GL_TRIANGLES;
    GLsizei vertexCount = 0;
};

Mesh3D CreateMesh3D(const std::vector<float>& data, GLenum primitive)
{
    Mesh3D m;
    m.primitive = primitive;
    m.vertexCount = static_cast<GLsizei>(data.size() / 8);

    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);

    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);

    // position: location 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // color: location 1
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // texcoord: location 2
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return m;
}

struct Mesh2D {
    GLuint VAO = 0;
    GLuint VBO = 0;
    GLenum primitive = GL_TRIANGLE_FAN;
    GLsizei vertexCount = 0;
};

Mesh2D CreateMesh2D(const std::vector<float>& data, GLenum primitive)
{
    Mesh2D m;
    m.primitive = primitive;
    m.vertexCount = static_cast<GLsizei>(data.size() / 2);

    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);

    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return m;
}

GLuint LoadTextureFromFile(const std::string& filename)
{
    sf::Image img;
    if (!img.loadFromFile(filename))
    {
        std::cout << "Failed to load texture: " << filename << std::endl;
        return 0;
    }
    img.flipVertically(); // чтобы совпали UV с OpenGL

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
        img.getSize().x, img.getSize().y,
        0, GL_RGBA, GL_UNSIGNED_BYTE, img.getPixelsPtr());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

int main()
{
    setlocale(LC_ALL, "ru");
    sf::Window window(
        sf::VideoMode({ 1200u, 900u }),
        "OpenGL lab: tetrahedron, textured cubes, gradient circle",
        sf::Style::Default
    );
    window.setFramerateLimit(60);
    window.setActive(true);

    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cout << "glewInit failed: "
            << reinterpret_cast<const char*>(glewGetErrorString(err))
            << std::endl;
        return 1;
    }

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << "\n";
    std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

    glEnable(GL_DEPTH_TEST);

    // --- компиляция шейдеров ---
    GLuint vert3D = CompileShader(GL_VERTEX_SHADER, vertex3DShaderSrc);
    GLuint fragTetra = CompileShader(GL_FRAGMENT_SHADER, fragTetraSrc);
    GLuint fragCubeColorTex = CompileShader(GL_FRAGMENT_SHADER, fragCubeColorTexSrc);
    GLuint fragCubeTwoTex = CompileShader(GL_FRAGMENT_SHADER, fragCubeTwoTexSrc);

    GLuint vertCircle = CompileShader(GL_VERTEX_SHADER, vertexCircleShaderSrc);
    GLuint fragCircle = CompileShader(GL_FRAGMENT_SHADER, fragCircleSrc);

    GLuint progTetra = LinkProgram(vert3D, fragTetra);
    GLuint progCubeColorTex = LinkProgram(vert3D, fragCubeColorTex);
    GLuint progCubeTwoTex = LinkProgram(vert3D, fragCubeTwoTex);

    GLuint progCircle = LinkProgram(vertCircle, fragCircle);

    glDeleteShader(vert3D);
    glDeleteShader(fragTetra);
    glDeleteShader(fragCubeColorTex);
    glDeleteShader(fragCubeTwoTex);

    glDeleteShader(vertCircle);
    glDeleteShader(fragCircle);

    // Поворот тетраэдра
    auto rotateTetra = [](float& x, float& y, float& z)
        {
            // поворот вокруг оси Y
            float angleY = 0.0f;
            float cY = cosf(angleY), sY = sinf(angleY);
            float x1 = cY * x + sY * z;
            float z1 = -sY * x + cY * z;

            // поворот вокруг оси X
            float angleX = 0.7f;
            float cX = cosf(angleX), sX = sinf(angleX);
            float y1 = cX * y - sX * z1;
            float z2 = sX * y + cX * z1;

            // лёгкий поворот вокруг Z, чтобы основание не было строго горизонтальным
            float angleZ = 0.4f;
            float cZ = cosf(angleZ), sZ = sinf(angleZ);
            float x2 = cZ * x1 - sZ * y1;
            float y2 = sZ * x1 + cZ * y1;

            x = x2;
            y = y2;
            z = z2;
        };

    // --- геометрия тетраэдра (4 вершины, 4 треугольника = 12 вершин) ---
    std::vector<float> tetraVerts = {
        // основание (треугольник)
        -0.3f, -0.3f,  0.2f,   1.0f, 0.0f, 0.0f,   0.f, 0.f,
         0.3f, -0.3f,  0.2f,   0.0f, 1.0f, 0.0f,   0.f, 0.f,
         0.0f, -0.3f, -0.3f,   0.0f, 0.0f, 1.0f,   0.f, 0.f,

         // боковая грань 1
         -0.3f, -0.3f,  0.2f,   1.0f, 0.0f, 0.0f,   0.f, 0.f,
          0.3f, -0.3f,  0.2f,   0.0f, 1.0f, 0.0f,   0.f, 0.f,
          0.0f,  0.4f,  0.0f,   1.0f, 1.0f, 0.0f,   0.f, 0.f,

          // боковая грань 2
           0.3f, -0.3f,  0.2f,   0.0f, 1.0f, 0.0f,   0.f, 0.f,
           0.0f, -0.3f, -0.3f,   0.0f, 0.0f, 1.0f,   0.f, 0.f,
           0.0f,  0.4f,  0.0f,   1.0f, 0.0f, 1.0f,   0.f, 0.f,

           // боковая грань 3
            0.0f, -0.3f, -0.3f,   0.0f, 0.0f, 1.0f,   0.f, 0.f,
           -0.3f, -0.3f,  0.2f,   1.0f, 0.0f, 0.0f,   0.f, 0.f,
            0.0f,  0.4f,  0.0f,   0.0f, 1.0f, 1.0f,   0.f, 0.f,
    };

    for (size_t i = 0; i < tetraVerts.size(); i += 8)
    {
        float& x = tetraVerts[i + 0];
        float& y = tetraVerts[i + 1];
        float& z = tetraVerts[i + 2];
        rotateTetra(x, y, z);
    }


    Mesh3D meshTetra = CreateMesh3D(tetraVerts, GL_TRIANGLES);


    // --- геометрия куба (общая для двух кубов) ---
    std::vector<float> cubeVerts;
    auto addFace = [&](float x1, float y1, float z1,
        float x2, float y2, float z2,
        float x3, float y3, float z3,
        float x4, float y4, float z4,
        float r, float g, float b)
        {
            // треугольник 1: v1,v2,v3
            cubeVerts.insert(cubeVerts.end(), {
                x1,y1,z1, r,g,b, 0.0f,0.0f,
                x2,y2,z2, r,g,b, 1.0f,0.0f,
                x3,y3,z3, r,g,b, 1.0f,1.0f,
                });
            // треугольник 2: v1,v3,v4
            cubeVerts.insert(cubeVerts.end(), {
                x1,y1,z1, r,g,b, 0.0f,0.0f,
                x3,y3,z3, r,g,b, 1.0f,1.0f,
                x4,y4,z4, r,g,b, 0.0f,1.0f,
                });
        };

    float s = 0.2f; // половина ребра куба
    // front (+Z)
    addFace(-s, -s, s, s, -s, s, s, s, s, -s, s, s, 1.0f, 0.0f, 0.0f);
    // back (-Z)
    addFace(s, -s, -s, -s, -s, -s, -s, s, -s, s, s, -s, 0.0f, 1.0f, 0.0f);
    // left (-X)
    addFace(-s, -s, -s, -s, -s, s, -s, s, s, -s, s, -s, 0.0f, 0.0f, 1.0f);
    // right (+X)
    addFace(s, -s, s, s, -s, -s, s, s, -s, s, s, s, 1.0f, 1.0f, 0.0f);
    // top (+Y)
    addFace(-s, s, s, s, s, s, s, s, -s, -s, s, -s, 1.0f, 0.0f, 1.0f);
    // bottom (-Y)
    addFace(-s, -s, -s, s, -s, -s, s, -s, s, -s, -s, s, 0.0f, 1.0f, 1.0f);

    // Поворот куба
    auto rotateCube = [](float& x, float& y, float& z)
        {
            float angleY = 0.7f;
            float cY = cosf(angleY), sY = sinf(angleY);
            float x1 = cY * x + sY * z;
            float z1 = -sY * x + cY * z;

            float angleX = -0.5f;
            float cX = cosf(angleX), sX = sinf(angleX);
            float y1 = cX * y - sX * z1;
            float z2 = sX * y + cX * z1;

            x = x1;
            y = y1;
            z = z2;
        };

    for (size_t i = 0; i < cubeVerts.size(); i += 8)
    {
        float& x = cubeVerts[i + 0];
        float& y = cubeVerts[i + 1];
        float& z = cubeVerts[i + 2];
        rotateCube(x, y, z);
    }
    Mesh3D meshCube = CreateMesh3D(cubeVerts, GL_TRIANGLES);

    // --- геометрия круга (2D triangle fan) ---
    std::vector<float> circleVerts;
    const int circleSegments = 100;
    float radius = 0.5f;

    // центр
    circleVerts.push_back(0.0f);
    circleVerts.push_back(0.0f);
    for (int i = 0; i <= circleSegments; ++i)
    {
        float t = (float)i / circleSegments;
        float angle = t * 2.0f * (float)M_PI;
        float x = cosf(angle) * radius;
        float y = sinf(angle) * radius;
        circleVerts.push_back(x);
        circleVerts.push_back(y);
    }
    Mesh2D meshCircle = CreateMesh2D(circleVerts, GL_TRIANGLE_FAN);

    // --- текстуры ---
    GLuint tex1 = LoadTextureFromFile("вазина.jpg");
    GLuint tex2 = LoadTextureFromFile("мопс.jpg");

    CheckOpenGLError("setup");

    // --- uniform locations ---
    // 3D
    GLint uOffsetTetra = glGetUniformLocation(progTetra, "uOffset");
    GLint uOffsetCubeColor = glGetUniformLocation(progCubeColorTex, "uOffset");
    GLint uOffsetCubeTwo = glGetUniformLocation(progCubeTwoTex, "uOffset");

    GLint uColorFactorLoc = glGetUniformLocation(progCubeColorTex, "uColorFactor");
    GLint uTexLoc = glGetUniformLocation(progCubeColorTex, "uTexture");

    GLint uTex1Loc = glGetUniformLocation(progCubeTwoTex, "uTexture1");
    GLint uTex2Loc = glGetUniformLocation(progCubeTwoTex, "uTexture2");
    GLint uMixFactorLoc = glGetUniformLocation(progCubeTwoTex, "uMixFactor");

    // circle
    GLint uScaleLoc = glGetUniformLocation(progCircle, "uScale");
    GLint uOffsetCircleLoc = glGetUniformLocation(progCircle, "uOffset");

    // --- параметры управления ---
    // позиции объектов
    float tetraOffset[3] = { -0.7f,  0.2f, 0.0f };
    float cubeColorOffset[3] = { 0.0f,  0.2f, 0.0f };
    float cubeTwoOffset[3] = { 0.7f,  0.2f, 0.0f };

    float circleScale[2] = { 0.6f, 0.6f };
    float circleOffset[2] = { 0.0f, -0.5f };

    float colorFactor = 0.5f; // uColorFactor для куба 1
    float mixFactor = 0.5f;   // uMixFactor для куба 2


    while (window.isOpen())
    {
        // обработка событий
        while (auto event = window.pollEvent())
        {
            // Закрытие окна
            if (event->is<sf::Event::Closed>())
                window.close();

            // Нажатие клавиши
            if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>())
            {
                sf::Keyboard::Key key = keyPressed->code;

                // Тетраэдр: движение по осям
                if (key == sf::Keyboard::Key::W) tetraOffset[1] += 0.05f;
                if (key == sf::Keyboard::Key::S) tetraOffset[1] -= 0.05f;
                if (key == sf::Keyboard::Key::A) tetraOffset[0] -= 0.05f;
                if (key == sf::Keyboard::Key::D) tetraOffset[0] += 0.05f;
                if (key == sf::Keyboard::Key::Q) tetraOffset[2] -= 0.05f;
                if (key == sf::Keyboard::Key::E) tetraOffset[2] += 0.05f;

                // Куб 1: влияние цвета
                if (key == sf::Keyboard::Key::Num1) colorFactor -= 0.05f;
                if (key == sf::Keyboard::Key::Num2) colorFactor += 0.05f;
                if (colorFactor < 0.0f) colorFactor = 0.0f;
                if (colorFactor > 1.0f) colorFactor = 1.0f;

                // Куб 2: смешивание текстур
                if (key == sf::Keyboard::Key::Num3) mixFactor -= 0.05f;
                if (key == sf::Keyboard::Key::Num4) mixFactor += 0.05f;
                if (mixFactor < 0.0f) mixFactor = 0.0f;
                if (mixFactor > 1.0f) mixFactor = 1.0f;

                // Круг: масштаб по осям
                if (key == sf::Keyboard::Key::Z) circleScale[0] -= 0.05f;
                if (key == sf::Keyboard::Key::X) circleScale[0] += 0.05f;
                if (key == sf::Keyboard::Key::C) circleScale[1] -= 0.05f;
                if (key == sf::Keyboard::Key::V) circleScale[1] += 0.05f;

                if (circleScale[0] < 0.1f) circleScale[0] = 0.1f;
                if (circleScale[1] < 0.1f) circleScale[1] = 0.1f;
                if (circleScale[0] > 2.0f) circleScale[0] = 2.0f;
                if (circleScale[1] > 2.0f) circleScale[1] = 2.0f;
            }
        }

        if (!window.isOpen()) break;
        window.setActive(true);

        glViewport(0, 0, window.getSize().x, window.getSize().y);
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ---------- РИСОВАНИЕ ТЕТРАЭДРА ----------
        glUseProgram(progTetra);
        if (uOffsetTetra >= 0)
            glUniform3f(uOffsetTetra, tetraOffset[0], tetraOffset[1], tetraOffset[2]);

        glBindVertexArray(meshTetra.VAO);
        glDrawArrays(meshTetra.primitive, 0, meshTetra.vertexCount);
        glBindVertexArray(0);
        glUseProgram(0);

        // ---------- РИСОВАНИЕ КУБА 1 (одна текстура + цвет) ----------
        glUseProgram(progCubeColorTex);
        if (uOffsetCubeColor >= 0)
            glUniform3f(uOffsetCubeColor, cubeColorOffset[0], cubeColorOffset[1], cubeColorOffset[2]);
        if (uColorFactorLoc >= 0)
            glUniform1f(uColorFactorLoc, colorFactor);

        if (uTexLoc >= 0)
            glUniform1i(uTexLoc, 0); // укажем, что sampler2D uTexture = GL_TEXTURE0

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex1);

        glBindVertexArray(meshCube.VAO);
        glDrawArrays(meshCube.primitive, 0, meshCube.vertexCount);
        glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);

        // ---------- РИСОВАНИЕ КУБА 2 (две текстуры) ----------
        glUseProgram(progCubeTwoTex);
        if (uOffsetCubeTwo >= 0)
            glUniform3f(uOffsetCubeTwo, cubeTwoOffset[0], cubeTwoOffset[1], cubeTwoOffset[2]);

        if (uTex1Loc >= 0) glUniform1i(uTex1Loc, 0);
        if (uTex2Loc >= 0) glUniform1i(uTex2Loc, 1);
        if (uMixFactorLoc >= 0) glUniform1f(uMixFactorLoc, mixFactor);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex2);

        glBindVertexArray(meshCube.VAO);
        glDrawArrays(meshCube.primitive, 0, meshCube.vertexCount);
        glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
        glUseProgram(0);

        // ---------- РИСОВАНИЕ ГРАДИЕНТНОГО КРУГА ----------
        glUseProgram(progCircle);
        if (uScaleLoc >= 0)
            glUniform2f(uScaleLoc, circleScale[0], circleScale[1]);
        if (uOffsetCircleLoc >= 0)
            glUniform2f(uOffsetCircleLoc, circleOffset[0], circleOffset[1]);

        glBindVertexArray(meshCircle.VAO);
        glDrawArrays(meshCircle.primitive, 0, meshCircle.vertexCount);
        glBindVertexArray(0);
        glUseProgram(0);

        CheckOpenGLError("draw");

        window.display();
    }

    // очистка
    glDeleteBuffers(1, &meshTetra.VBO);
    glDeleteVertexArrays(1, &meshTetra.VAO);

    glDeleteBuffers(1, &meshCube.VBO);
    glDeleteVertexArrays(1, &meshCube.VAO);

    glDeleteBuffers(1, &meshCircle.VBO);
    glDeleteVertexArrays(1, &meshCircle.VAO);

    glDeleteTextures(1, &tex1);
    glDeleteTextures(1, &tex2);

    glDeleteProgram(progTetra);
    glDeleteProgram(progCubeColorTex);
    glDeleteProgram(progCubeTwoTex);
    glDeleteProgram(progCircle);

    return 0;
}