// main.cpp
#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
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

// Вершинный шейдер (один, используемый всеми программами)
const char* vertexShaderSrc = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor; // может не использоваться в некоторых фраг.шейдерах

    out vec3 vColor;
    uniform vec2 uOffset; // сдвиг фигуры по экрану

    void main()
    {
        vColor = aColor;
        vec2 pos = aPos.xy + uOffset;
        gl_Position = vec4(pos, aPos.z, 1.0);
    }
)";

// Фрагментный шейдер 1: константный цвет, задан в коде шейдера
const char* fragConstSrc = R"(
    #version 330 core
    out vec4 FragColor;
    void main()
    {
        // Поменяй здесь цвет, если нужно (константа в шейдере)
        FragColor = vec4(0.0, 1.0, 0.0, 1.0); // зелёный
    }
)";

// Фрагментный шейдер 2: цвет через uniform
const char* fragUniformSrc = R"(
    #version 330 core
    out vec4 FragColor;
    uniform vec4 uColor;
    void main()
    {
        FragColor = uColor;
    }
)";

// Фрагментный шейдер 3: градиент — берёт цвет из вершин
const char* fragVertexColorSrc = R"(
    #version 330 core
    in vec3 vColor;
    out vec4 FragColor;
    void main()
    {
        FragColor = vec4(vColor, 1.0);
    }
)";

// Утилита: создаёт VAO для массива вершин (position + color interleaved)
struct Mesh {
    GLuint VAO = 0;
    GLuint VBO = 0;
    GLenum primitive = GL_TRIANGLES;
    GLsizei vertexCount = 0;
};

Mesh CreateMesh(const std::vector<float>& interleavedPosColor, GLenum primitive)
{
    Mesh m;
    m.primitive = primitive;
    m.vertexCount = static_cast<GLsizei>(interleavedPosColor.size() / 6); // 3 pos + 3 color
    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);

    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, interleavedPosColor.size() * sizeof(float), interleavedPosColor.data(), GL_STATIC_DRAW);

    // позиция: location = 0 (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // цвет: location = 1 (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return m;
}

int main()
{
    sf::Window window(
        sf::VideoMode({ 1200u, 1000u }),
        "OpenGL: quad, fan, pentagon — const/uniform/vertex colors",
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

    // Компилируем шейдеры и программы
    GLuint vert = CompileShader(GL_VERTEX_SHADER, vertexShaderSrc);

    GLuint fragConst = CompileShader(GL_FRAGMENT_SHADER, fragConstSrc);
    GLuint progConst = LinkProgram(vert, fragConst);

    GLuint fragUniform = CompileShader(GL_FRAGMENT_SHADER, fragUniformSrc);
    GLuint progUniform = LinkProgram(vert, fragUniform);

    GLuint fragVertexColor = CompileShader(GL_FRAGMENT_SHADER, fragVertexColorSrc);
    GLuint progVertexColor = LinkProgram(vert, fragVertexColor);

    // стираем шейдеры после линковки
    glDeleteShader(vert);
    glDeleteShader(fragConst);
    glDeleteShader(fragUniform);
    glDeleteShader(fragVertexColor);

    // Создаём данные вершин (position + color interleaved)
    // 1) Четырёхугольник: два треугольника (6 вершин)
    //    центр фигуры в начале (0,0) — будем смещать uOffset-ом при рисовании
    std::vector<float> quad = {
        // x, y, z,    r, g, b
        -0.25f, -0.25f, 0.0f,  1.0f, 0.0f, 0.0f, // левый нижний (красный)
         0.25f, -0.25f, 0.0f,  0.0f, 1.0f, 0.0f, // правый нижний (зелёный)
         0.25f,  0.25f, 0.0f,  0.0f, 0.0f, 1.0f, // правый верхний (синий)

        -0.25f, -0.25f, 0.0f,  1.0f, 0.0f, 0.0f, // левый нижний
         0.25f,  0.25f, 0.0f,  0.0f, 0.0f, 1.0f, // правый верхний
        -0.25f,  0.25f, 0.0f,  1.0f, 1.0f, 0.0f  // левый верхний (жёлтый)
    };

    // 2) Веер: triangle fan (центр + несколько вершин дугой)
    //    центр (0,0)
    const int fanSegments = 5; // количество "лопастей" — можно менять
    std::vector<float> fan;
    // центр
    fan.push_back(-0.0f); fan.push_back(0.0f); fan.push_back(0.0f);
    fan.push_back(0.0f); fan.push_back(1.0f); fan.push_back(0.0f); // белый центр
    float radius = 0.4f;
    float startAngle = -M_PI / 4.0f;
    float sweep = M_PI / 3.0f; // ширина веера
    for (int i = 0; i <= fanSegments; ++i)
    {
        float t = (float)i / fanSegments; // 0 → 1
        float ang = startAngle + t * sweep;

        float x = cosf(ang) * radius;
        float y = sinf(ang) * radius;

        fan.push_back(x);
        fan.push_back(y);
        fan.push_back(0.0f); // Z

        // ---- конкретные цвета ----
        float cr, cg, cb;

        if (i == 0) {
            cr = 1.0f; cg = 0.0f; cb = 0.0f; // красный
        }
        else if (i == fanSegments) {
            cr = 0.0f; cg = 0.0f; cb = 1.0f; // синий
        }
        else {
            cr = 1.0f - t; // красный уменьшается
            cg = 0.0f;     // зелёный = 0
            cb = t;        // синий растёт
        }

        fan.push_back(cr);
        fan.push_back(cg);
        fan.push_back(cb);
    }

    // 3) Правильный пятиугольник: используем triangle fan (центр + 5 вершин)
    std::vector<float> pent;
    std::vector<std::array<float, 3>> colors = {
        {1.0f, 0.0f, 0.0f}, // вершина 1 - красная
        {0.0f, 1.0f, 0.0f}, // вершина 2 - зелёная
        {0.0f, 0.0f, 1.0f}, // вершина 3 - синяя
        {1.0f, 1.0f, 0.0f}, // вершина 4 - жёлтая
        {1.0f, 0.0f, 1.0f}, // вершина 5 - фиолетовая
        {1.0f, 0.0f, 0.0f}  // последняя вершина повторяет первую для замыкания
    };
    // центр
    float cr_center = (colors[0][0] + colors[1][0] + colors[2][0] + colors[3][0] + colors[4][0]) / 5;
    float cg_center = (colors[0][1] + colors[1][1] + colors[2][1] + colors[3][1] + colors[4][1]) / 5;
    float cb_center = (colors[0][2] + colors[1][2] + colors[2][2] + colors[3][2] + colors[4][2]) / 5;
    pent.push_back(0.0f); pent.push_back(0.0f); pent.push_back(0.0f);
    pent.push_back(cr_center); pent.push_back(cg_center); pent.push_back(cb_center);
    const int pSides = 5;
    float pRadius = 0.3f;

    for (int i = 0; i <= pSides; ++i)
    {
        float ang = (float)i / pSides * 2.0f * M_PI + (M_PI / 2.0f);
        float x = cosf(ang) * pRadius;
        float y = sinf(ang) * pRadius;

        pent.push_back(x);
        pent.push_back(y);
        pent.push_back(0.0f);

        pent.push_back(colors[i][0]);
        pent.push_back(colors[i][1]);
        pent.push_back(colors[i][2]);
    }

    // Создаём mesh-ы
    Mesh meshQuad = CreateMesh(quad, GL_TRIANGLES); // glDrawArrays(GL_TRIANGLES, 0, 6)
    Mesh meshFan = CreateMesh(fan, GL_TRIANGLE_FAN); // glDrawArrays(GL_TRIANGLE_FAN, 0, fanCount)
    Mesh meshPent = CreateMesh(pent, GL_TRIANGLE_FAN); // glDrawArrays(GL_TRIANGLE_FAN, 0, pentCount)

    CheckOpenGLError("setup");

    // Находим uniform-переменные
    GLint uniOffsetConst = glGetUniformLocation(progConst, "uOffset");
    GLint uniOffsetUniform = glGetUniformLocation(progUniform, "uOffset");
    GLint uniOffsetVertex = glGetUniformLocation(progVertexColor, "uOffset");

    GLint uniColor = glGetUniformLocation(progUniform, "uColor"); // для программы с uniform

    // Основной цикл
    while (window.isOpen())
    {
        // события
        while (const std::optional<sf::Event> event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }
        if (!window.isOpen()) break;
        window.setActive(true);

        glViewport(0, 0, window.getSize().x, window.getSize().y);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Рисуем 3 колонки: x = -0.65 (константа), 0.0 (uniform), +0.65 (вершинные цвета)
        const float offsetsX[3] = { -0.65f, 0.0f, 0.65f };

        // --- Четырёхугольник ---
        // 1) константой (в шейдере)
        glUseProgram(progConst);
        if (uniOffsetConst >= 0) glUniform2f(uniOffsetConst, offsetsX[0], 0.7f);
        glBindVertexArray(meshQuad.VAO);
        // ВАЖНО: здесь демонстрация изменения параметра glDrawArrays:
        // для квадрата мы используем GL_TRIANGLES и 6 вершин:
        glDrawArrays(meshQuad.primitive, 0, meshQuad.vertexCount); // <<< ЗДЕСЬ: GL_TRIANGLES, count = 6
        glBindVertexArray(0);
        glUseProgram(0);

        // 2) uniform (цвет передан из программы)
        glUseProgram(progUniform);
        if (uniOffsetUniform >= 0) glUniform2f(uniOffsetUniform, offsetsX[1], 0.7f);
        if (uniColor >= 0) glUniform4f(uniColor, 0.7f, 0.7f, 0.7f, 1.0f); // пример цвета через uniform
        glBindVertexArray(meshQuad.VAO);
        glDrawArrays(meshQuad.primitive, 0, meshQuad.vertexCount); // GL_TRIANGLES, 6
        glBindVertexArray(0);
        glUseProgram(0);

        // 3) градиент (цвет с вершин)
        glUseProgram(progVertexColor);
        if (uniOffsetVertex >= 0) glUniform2f(uniOffsetVertex, offsetsX[2], 0.7f);
        glBindVertexArray(meshQuad.VAO);
        glDrawArrays(meshQuad.primitive, 0, meshQuad.vertexCount); // GL_TRIANGLES, 6
        glBindVertexArray(0);
        glUseProgram(0);

        // --- Веер (triangle fan) ---
        // NOTE: здесь используем GL_TRIANGLE_FAN и количество вершин = meshFan.vertexCount
        // 1) const
        glUseProgram(progConst);
        if (uniOffsetConst >= 0) glUniform2f(uniOffsetConst, offsetsX[0] - 0.1f, 0.0f);
        glBindVertexArray(meshFan.VAO);
        glDrawArrays(meshFan.primitive, 0, meshFan.vertexCount); // <<< GL_TRIANGLE_FAN, count = fan vertex count
        glBindVertexArray(0);
        glUseProgram(0);

        // 2) uniform
        glUseProgram(progUniform);
        if (uniOffsetUniform >= 0) glUniform2f(uniOffsetUniform, offsetsX[1] - 0.1f, 0.0f);
        if (uniColor >= 0) glUniform4f(uniColor, 1.0f, 0.5f, 0.0f, 1.0f); // оранжевый
        glBindVertexArray(meshFan.VAO);
        glDrawArrays(meshFan.primitive, 0, meshFan.vertexCount);
        glBindVertexArray(0);
        glUseProgram(0);

        // 3) vertex colors
        glUseProgram(progVertexColor);
        if (uniOffsetVertex >= 0) glUniform2f(uniOffsetVertex, offsetsX[2] - 0.1f, 0.0f);
        glBindVertexArray(meshFan.VAO);
        glDrawArrays(meshFan.primitive, 0, meshFan.vertexCount);
        glBindVertexArray(0);
        glUseProgram(0);

        // --- Пятиугольник (triangle fan) ---
        // 1) const
        glUseProgram(progConst);
        if (uniOffsetConst >= 0) glUniform2f(uniOffsetConst, offsetsX[0], -0.7f);
        glBindVertexArray(meshPent.VAO);
        glDrawArrays(meshPent.primitive, 0, meshPent.vertexCount); // <<< GL_TRIANGLE_FAN, count = pent vertex count
        glBindVertexArray(0);
        glUseProgram(0);

        // 2) uniform
        glUseProgram(progUniform);
        if (uniOffsetUniform >= 0) glUniform2f(uniOffsetUniform, offsetsX[1], -0.7f);
        if (uniColor >= 0) glUniform4f(uniColor, 0.2f, 0.4f, 1.0f, 1.0f); // синий-ish
        glBindVertexArray(meshPent.VAO);
        glDrawArrays(meshPent.primitive, 0, meshPent.vertexCount);
        glBindVertexArray(0);
        glUseProgram(0);

        // 3) vertex colors
        glUseProgram(progVertexColor);
        if (uniOffsetVertex >= 0) glUniform2f(uniOffsetVertex, offsetsX[2], -0.7f);
        glBindVertexArray(meshPent.VAO);
        glDrawArrays(meshPent.primitive, 0, meshPent.vertexCount);
        glBindVertexArray(0);
        glUseProgram(0);

        CheckOpenGLError("draw");

        window.display();
    }

    // Очистка
    glDeleteBuffers(1, &meshQuad.VBO);
    glDeleteVertexArrays(1, &meshQuad.VAO);
    glDeleteBuffers(1, &meshFan.VBO);
    glDeleteVertexArrays(1, &meshFan.VAO);
    glDeleteBuffers(1, &meshPent.VBO);
    glDeleteVertexArrays(1, &meshPent.VAO);

    glDeleteProgram(progConst);
    glDeleteProgram(progUniform);
    glDeleteProgram(progVertexColor);

    return 0;
}