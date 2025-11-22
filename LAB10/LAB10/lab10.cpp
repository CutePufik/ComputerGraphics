#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <iostream>
#include <optional>
#include <vector>

// Лог компиляции шейдера (из задания)
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

// Проверка общих ошибок OpenGL
void CheckOpenGLError(const char* where)
{
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR)
    {
        std::cout << "OpenGL error at " << where << ": 0x"
            << std::hex << err << std::dec << std::endl;
    }
}

int main()
{
    // Создаём окно c контекстом OpenGL
    sf::ContextSettings settings;
    settings.majorVersion = 3;     // попросим 3.3
    settings.minorVersion = 3;
    settings.depthBits = 24;

    sf::Window window(
        sf::VideoMode({ 800u, 600u }),
        "Green Triangle (OpenGL + SFML 3)",
        sf::State::Windowed,
        settings
    );

    window.setFramerateLimit(60);

    // Делаем контекст активным и инициализируем GLEW
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

    // ---------- ШЕЙДЕРЫ ----------

    const char* vertexShaderSrc = R"(
        #version 330 core

        layout (location = 0) in vec3 aPos;

        void main()
        {
            gl_Position = vec4(aPos, 1.0);
        }
    )";

    const char* fragmentShaderSrc = R"(
        #version 330 core

        out vec4 FragColor;

        void main()
        {
            // зелёный треугольник
            FragColor = vec4(0.0, 1.0, 0.0, 1.0);
        }
    )";

    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vertexShaderSrc, nullptr);
    glCompileShader(vertShader);
    ShaderLog(vertShader);

    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragmentShaderSrc, nullptr);
    glCompileShader(fragShader);
    ShaderLog(fragShader);

    // ---------- ШЕЙДЕРНАЯ ПРОГРАММА ----------

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);

    // ошибки линковки программы
    GLint success = 0;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        GLint logLen = 0;
        glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &logLen);
        std::vector<char> log(logLen);
        glGetProgramInfoLog(shaderProgram, logLen, nullptr, log.data());
        std::cout << "Program link error:\n" << log.data() << std::endl;
        return 1;
    }

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    // ---------- VBO (+ VAO) ----------

    float triangleVertices[] = {
        // x,    y,    z
        -0.5f, -0.5f, 0.0f,   // левый нижний
         0.5f, -0.5f, 0.0f,   // правый нижний
         0.0f,  0.5f, 0.0f    // верхний
    };

    GLuint VBO = 0;
    GLuint VAO = 0;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,
        sizeof(triangleVertices),
        triangleVertices,
        GL_STATIC_DRAW);

    // Атрибут позиции (location = 0)
    glVertexAttribPointer(
        0,                 // индекс атрибута
        3,                 // по 3 float на вершину
        GL_FLOAT,
        GL_FALSE,
        3 * sizeof(float), // шаг
        (void*)0
    );
    glEnableVertexAttribArray(0);

    // Можно разбиндить
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    CheckOpenGLError("setup");

    // ----------- ГЛАВНЫЙ ЦИКЛ -----------

    while (window.isOpen())
    {
        // обработка событий
        while (const std::optional<sf::Event> event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        // делаем контекст активным (на всякий случай)
        window.setActive(true);

        // очистка экрана (чёрный фон)
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        // рисуем треугольник
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);

        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindVertexArray(0);
        glUseProgram(0);

        CheckOpenGLError("draw");

        window.display();
    }

    // ----------- ОЧИСТКА РЕСУРСОВ -----------

    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shaderProgram);

    return 0;
}