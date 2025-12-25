#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <ctime>

#include "utils.h"
#include "shaders.h"

// ======================= ИГРОВЫЕ СТРУКТУРЫ =======================
struct RenderObject
{
    Mesh* mesh = nullptr;
    GLuint texDiffuse = 0;
    GLuint texNormal = 0;
    GLuint texLight = 0;

    bool useNormal = false;
    bool useLight = false;
    bool isCloud = false;

    Vec3 pos{ 0,0,0 };
    Vec3 rot{ 0,0,0 };
    Vec3 scale{ 1,1,1 };

    float collisionRadius = 1.0f;
    bool alive = true;
};

struct Package
{
    Vec3 pos{ 0,0,0 };
    Vec3 vel{ 0,-10,0 };
    float radius = 0.4f;
    bool alive = true;
};

// ======================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (СПЕЦИФИЧНЫЕ ДЛЯ ИГРЫ) =======================
void PlaceOnGround(RenderObject& o, float groundY = 0.0f, float extra = 0.0f)
{
    if (!o.mesh) return;
    o.pos.y = groundY - o.mesh->minY * o.scale.y + extra;
}

// матрица объекта (yaw + pitch + scale)
Mat4 BuildModelMatrix(const RenderObject& o)
{
    Mat4 T = Mat4::Translation(o.pos.x, o.pos.y, o.pos.z);
    Mat4 Ry = Mat4::RotationY(o.rot.y);
    Mat4 Rx = Mat4::RotationX(o.rot.x);
    Mat4 S = Mat4::Scale(o.scale.x, o.scale.y, o.scale.z);
    return T * Ry * Rx * S;
}

// ======================= MAIN =======================
int main()
{
    setlocale(LC_ALL, "ru_RU.utf8");
    srand((unsigned)time(nullptr));

    // Создание окна
    sf::Window window(
        sf::VideoMode({ 1200u, 900u }),
        "Airship Delivery (OpenGL 3.3, SFML)",
        sf::Style::Default
    );
    window.setFramerateLimit(60);
    window.setActive(true);

    // Инициализация GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cout << "glewInit failed: " << (const char*)glewGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "OpenGL: " << glGetString(GL_VERSION) << "\n";
    std::cout << "GLSL:   " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

    // Настройка OpenGL
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // --- Шейдеры ---
    GLuint vs = CompileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint prog = LinkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);

    // Получение uniform-локаций
    GLint uModelLoc = glGetUniformLocation(prog, "uModel");
    GLint uViewLoc = glGetUniformLocation(prog, "uView");
    GLint uProjLoc = glGetUniformLocation(prog, "uProj");

    GLint uDiffuseLoc = glGetUniformLocation(prog, "uDiffuse");
    GLint uNormalLoc = glGetUniformLocation(prog, "uNormalMap");
    GLint uLightLoc = glGetUniformLocation(prog, "uLightMap");

    GLint uUseNormalLoc = glGetUniformLocation(prog, "uUseNormalMap");
    GLint uUseLightmapLoc = glGetUniformLocation(prog, "uUseLightMap");

    GLint uLightDirLoc = glGetUniformLocation(prog, "uLightDir");
    GLint uLightColorLoc = glGetUniformLocation(prog, "uLightColor");
    GLint uAmbientLoc = glGetUniformLocation(prog, "uAmbientColor");

    // Uniforms для прожектора
    GLint uSpotlightEnabledLoc = glGetUniformLocation(prog, "uSpotlightEnabled");
    GLint uSpotlightPosLoc = glGetUniformLocation(prog, "uSpotlightPos");
    GLint uSpotlightDirLoc = glGetUniformLocation(prog, "uSpotlightDir");
    GLint uSpotlightColorLoc = glGetUniformLocation(prog, "uSpotlightColor");
    GLint uSpotlightCutoffLoc = glGetUniformLocation(prog, "uSpotlightCutoff");
    GLint uSpotlightOuterCutoffLoc = glGetUniformLocation(prog, "uSpotlightOuterCutoff");
    GLint uSpotlightConstantLoc = glGetUniformLocation(prog, "uSpotlightConstant");
    GLint uSpotlightLinearLoc = glGetUniformLocation(prog, "uSpotlightLinear");
    GLint uSpotlightQuadraticLoc = glGetUniformLocation(prog, "uSpotlightQuadratic");

    GLint uIsCloudLoc = glGetUniformLocation(prog, "uIsCloud");

    // Функция создания проекционной матрицы
    auto makeProjection = [&](unsigned w, unsigned h)
        {
            float aspect = (h == 0) ? 1.0f : (float)w / (float)h;
            return Mat4::Perspective(deg2rad(60.0f), aspect, 0.1f, 1000.0f);
        };
    Mat4 proj = makeProjection(window.getSize().x, window.getSize().y);

    // ======================= ЗАГРУЗКА МЕШЕЙ =======================
    // Ground
    Mesh groundMesh = CreateGround(80.0f, 20.0f);

    // Загрузка моделей
    std::vector<VertexPNUt> airshipVerts;
    if (!LoadOBJ_PNU("assets/airship.obj", airshipVerts)) return 1;
    Mesh airshipMesh = CreateMesh_PNUt(airshipVerts);

    std::vector<VertexPNUt> treeVerts;
    if (!LoadOBJ_PNU("assets/tree.obj", treeVerts)) return 1;
    Mesh treeMesh = CreateMesh_PNUt(treeVerts);

    std::vector<VertexPNUt> houseVerts;
    if (!LoadOBJ_PNU("assets/house.obj", houseVerts)) return 1;
    Mesh houseMesh = CreateMesh_PNUt(houseVerts);

    std::vector<VertexPNUt> deco1Verts;
    if (!LoadOBJ_PNU("assets/Hamburger.obj", deco1Verts)) return 1;
    Mesh deco1Mesh = CreateMesh_PNUt(deco1Verts);

    std::vector<VertexPNUt> deco2Verts;
    if (!LoadOBJ_PNU("assets/deco2.obj", deco2Verts)) return 1;
    Mesh deco2Mesh = CreateMesh_PNUt(deco2Verts);

    // ======================= ЗАГРУЗКА ТЕКСТУР =======================
    GLuint groundTex = LoadTextureFromFile("assets/ground_diffuse.png", true);
    if (!groundTex) {
        std::cout << "WARN: no ground texture. Put assets/ground_diffuse.png\n";
    }
    std::cout << "groundTex id = " << groundTex << "\n";

    GLuint airshipDiffuse = LoadTextureFromFile("assets/airship_diffuse.png", true);
    GLuint airshipNormal = LoadTextureFromFile("assets/airship_normal.png", false);
    if (!airshipDiffuse || !airshipNormal)
    {
        std::cout << "Need airship_diffuse + airship_normal\n";
        return 1;
    }

    GLuint treeDiffuse = LoadTextureFromFile("assets/tree_diffuse.png", true);
    if (!treeDiffuse) return 1;

    GLuint houseDiffuse = LoadTextureFromFile("assets/house_diffuse.png", true);
    GLuint houseLight = LoadTextureFromFile("assets/house_lightmap.png", false);
    if (!houseDiffuse || !houseLight)
    {
        std::cout << "Need house_diffuse + house_lightmap\n";
        return 1;
    }

    GLuint deco1Tex = LoadTextureFromFile("assets/Hamburger_BaseColor.png", true);
    GLuint deco2Tex = LoadTextureFromFile("assets/deco2_diffuse.png", true);
    if (!deco1Tex || !deco2Tex) return 1;

    // ======================= ОБЛАКА И ВОЗДУШНЫЕ ШАРЫ =======================
    Mesh cloudMesh;
    GLuint cloudTex = 0;
    std::vector<VertexPNUt> cloudVerts;

    if (LoadOBJ_PNU("assets/cloud.obj", cloudVerts)) {
        cloudMesh = CreateMesh_PNUt(cloudVerts);
        cloudTex = LoadTextureFromFile("assets/cloud.png", true);
        if (!cloudTex) {
            std::cout << "Cloud texture not found, using deco1 texture\n";
            cloudTex = deco1Tex;
        }
    }
    else {
        std::cout << "Cloud model not found, using deco1 as cloud\n";
        cloudMesh = deco1Mesh;
        cloudTex = deco1Tex;
    }

    Mesh balloonMesh;
    GLuint balloonTex = 0;
    std::vector<VertexPNUt> balloonVerts;

    if (LoadOBJ_PNU("assets/balloon.obj", balloonVerts)) {
        balloonMesh = CreateMesh_PNUt(balloonVerts);
        balloonTex = LoadTextureFromFile("assets/balloon.png", true);
        if (!balloonTex) {
            std::cout << "Balloon texture not found";
            return -1;
        }
    }
    else {
        std::cout << "Balloon model not found";
        return -1;
    }

    std::vector<RenderObject> clouds;
    std::vector<RenderObject> balloons;

    // ======================= ОСНОВНЫЕ ОБЪЕКТЫ СЦЕНЫ =======================
    RenderObject ground;
    ground.mesh = &groundMesh;
    ground.texDiffuse = groundTex;
    ground.useNormal = false;
    ground.useLight = false;
    ground.scale = { 1,1,1 };
    ground.pos = { 0, 0, 0 };
    ground.collisionRadius = 99999.0f;

    RenderObject tree;
    tree.mesh = &treeMesh;
    tree.texDiffuse = treeDiffuse;
    tree.pos = { 0,0,0 };
    tree.scale = { 0.005f, 0.005f, 0.005f };
    PlaceOnGround(tree, 0.0f);
    tree.collisionRadius = 2.5f;

    RenderObject airship;
    airship.mesh = &airshipMesh;
    airship.texDiffuse = airshipDiffuse;
    airship.texNormal = airshipNormal;
    airship.useNormal = true;
    airship.useLight = false;
    airship.pos = { 0, 12.0f, 20.0f };
    airship.scale = { 0.0005f,0.0005f,0.0005f };
    airship.rot = { 0.0f, 0.0f, 0.0f };
    airship.collisionRadius = 2.0f;

    std::vector<RenderObject> targets;
    std::vector<RenderObject> decos;

    // Функция генерации небесных объектов
    auto generateSkyObjects = [&]() {
        clouds.clear();
        balloons.clear();

        // Генерация облаков
        int cloudCount = 5 + rand() % 3;
        for (int i = 0; i < cloudCount; i++) {
            RenderObject cloud;
            cloud.mesh = &cloudMesh;
            cloud.texDiffuse = cloudTex;
            cloud.useNormal = false;
            cloud.useLight = false;
            cloud.isCloud = true;

            float x = frand(-70.0f, 70.0f);
            float y = frand(15.0f, 35.0f);
            float z = frand(-70.0f, 70.0f);
            cloud.pos = { x, y, z };

            float scale = frand(3.0f, 8.0f);
            cloud.scale = { scale, scale * 0.7f, scale };
            cloud.rot.y = frand(0.0f, (float)M_PI * 2.0f);
            cloud.rot.x = frand(-0.2f, 0.2f);

            cloud.alive = true;
            cloud.collisionRadius = scale * 0.5f;

            clouds.push_back(cloud);
        }

        // Генерация воздушных шаров
        int balloonCount = 3 + rand() % 4;
        for (int i = 0; i < balloonCount; i++) {
            RenderObject balloon;
            balloon.mesh = &balloonMesh;
            balloon.texDiffuse = balloonTex;
            balloon.useNormal = false;
            balloon.useLight = false;
            balloon.isCloud = false;

            float x = frand(-60.0f, 60.0f);
            float y = frand(10.0f, 25.0f);
            float z = frand(-60.0f, 60.0f);
            balloon.pos = { x, y, z };

            float scale = frand(0.5f, 1.5f);
            balloon.scale = { scale, scale, scale };
            balloon.rot.y = frand(0.0f, (float)M_PI * 2.0f);

            balloon.alive = true;
            balloon.collisionRadius = scale * 0.8f;

            balloons.push_back(balloon);
        }

        std::cout << "Generated " << clouds.size() << " clouds and "
            << balloons.size() << " balloons\n";
        };

    // Функция регенерации всей сцены
    auto regenerateScene = [&]()
        {
            targets.clear();
            decos.clear();

            // Создание целей (домов)
            int targetCount = 7;
            for (int i = 0; i < targetCount; i++)
            {
                RenderObject h;
                h.mesh = &houseMesh;
                h.texDiffuse = houseDiffuse;
                h.texLight = houseLight;
                h.useLight = true;
                h.useNormal = false;
                h.alive = true;

                for (int tries = 0; tries < 100; ++tries)
                {
                    float x = frand(-55.0f, 55.0f);
                    float z = frand(-55.0f, 55.0f);
                    Vec3 p = { x, 0.0f, z };
                    if (Length(p - tree.pos) > 8.0f)
                    {
                        h.pos = p;
                        break;
                    }
                }

                h.scale = { 2.0f,2.0f,2.0f };
                PlaceOnGround(h, 0.0f);
                h.rot.y = frand(0.0f, (float)M_PI * 2.0f);
                h.collisionRadius = 2.2f;

                targets.push_back(h);
            }

            // Декорации 1 типа
            for (int i = 0; i < 10; i++)
            {
                RenderObject d;
                d.mesh = &deco1Mesh;
                d.texDiffuse = deco1Tex;
                d.scale = { frand(1.0f, 2.5f), frand(1.0f, 2.5f), frand(1.0f, 2.5f) };
                d.rot.y = frand(0.0f, (float)M_PI * 2.0f);

                float x = frand(-60.0f, 60.0f);
                float z = frand(-60.0f, 60.0f);
                d.pos = { x, 0.0f, z };
                PlaceOnGround(d, 0.0f);
                if (Length(d.pos - tree.pos) < 6.0f) d.pos = d.pos + Vec3(8, 0, 8);

                d.alive = true;
                d.collisionRadius = 0.0f;
                decos.push_back(d);
            }

            // Декорации 2 типа
            for (int i = 0; i < 10; i++)
            {
                RenderObject d;
                d.mesh = &deco2Mesh;
                d.texDiffuse = deco2Tex;
                d.scale = { frand(1.0f, 2.5f), frand(1.0f, 2.5f), frand(1.0f, 2.5f) };
                d.rot.y = frand(0.0f, (float)M_PI * 2.0f);

                float x = frand(-60.0f, 60.0f);
                float z = frand(-60.0f, 60.0f);
                d.pos = { x, 0.0f, z };
                PlaceOnGround(d, 0.0f);
                if (Length(d.pos - tree.pos) < 6.0f) d.pos = d.pos + Vec3(-8, 0, 8);

                d.alive = true;
                d.collisionRadius = 0.0f;
                decos.push_back(d);
            }

            // Генерация облаков и шаров
            generateSkyObjects();

            std::cout << "Scene regenerated: targets=" << targets.size()
                << ", decos=" << decos.size()
                << ", clouds=" << clouds.size()
                << ", balloons=" << balloons.size() << "\n";
        };

    regenerateScene();

    // ======================= ПАКЕТЫ И СЧЕТЧИКИ =======================
    std::vector<Package> packages;
    int hits = 0;

    // ======================= КАМЕРА И УПРАВЛЕНИЕ =======================
    Vec3 worldUp(0, 1, 0);
    float camYaw = 0.0f;
    float camPitch = deg2rad(-15.0f);
    float camDist = 18.0f;
    float camHeight = 8.0f;

    bool aimingMode = false;
    float aimCamHeight = 5.0f;
    float aimCamDist = 1.0f;

    // ======================= ПРОЖЕКТОР =======================
    bool spotlightEnabled = false;
    bool prevSpotlightToggle = false;

    float spotlightCutoff = std::cos(deg2rad(12.5f));
    float spotlightOuterCutoff = std::cos(deg2rad(17.5f));
    float spotlightConstant = 1.0f;
    float spotlightLinear = 0.09f;
    float spotlightQuadratic = 0.032f;
    Vec3 spotlightColor = { 1.0f, 1.0f, 0.9f };

    // ======================= ГЛАВНЫЙ ЦИКЛ =======================
    sf::Clock clock;
    float moveSpeed = 20.0f;
    float rotSpeed = deg2rad(80.0f);

    bool prevDrop = false;
    bool prevR = false;

    // Функция отрисовки объекта
    auto drawObj = [&](const RenderObject& o)
        {
            if (!o.alive) return;
            if (!o.mesh) return;

            Mat4 model = BuildModelMatrix(o);
            glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, model.m);

            glUniform1i(uUseNormalLoc, o.useNormal ? 1 : 0);
            glUniform1i(uUseLightmapLoc, o.useLight ? 1 : 0);
            glUniform1i(uIsCloudLoc, o.isCloud ? 1 : 0);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, o.texDiffuse);
            glUniform1i(uDiffuseLoc, 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, o.texNormal ? o.texNormal : 0);
            glUniform1i(uNormalLoc, 1);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, o.texLight ? o.texLight : 0);
            glUniform1i(uLightLoc, 2);

            glBindVertexArray(o.mesh->VAO);
            glDrawArrays(GL_TRIANGLES, 0, o.mesh->vertexCount);
            glBindVertexArray(0);
        };

    while (window.isOpen())
    {
        float dt = clock.restart().asSeconds();

        // Обработка событий
        while (auto ev = window.pollEvent())
        {
            if (ev->is<sf::Event::Closed>()) window.close();

            if (const auto* resized = ev->getIf<sf::Event::Resized>())
            {
                glViewport(0, 0, resized->size.x, resized->size.y);
                proj = makeProjection(resized->size.x, resized->size.y);
            }

            if (const auto* keyPressed = ev->getIf<sf::Event::KeyPressed>())
            {
                if (keyPressed->code == sf::Keyboard::Key::V)
                {
                    aimingMode = !aimingMode;
                    std::cout << "Aiming mode: " << (aimingMode ? "ON" : "OFF") << std::endl;
                }
            }
        }

        // ====== Управление дирижаблем ======
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left))
            airship.rot.y -= rotSpeed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right))
            airship.rot.y += rotSpeed * dt;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up))
            camPitch += deg2rad(30.0f) * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down))
            camPitch -= deg2rad(30.0f) * dt;

        camPitch = std::clamp(camPitch, deg2rad(-60.0f), deg2rad(-5.0f));

        Vec3 forward = { std::sin(airship.rot.y), 0.0f, std::cos(airship.rot.y) };
        forward = Normalize(forward);
        Vec3 right = Normalize(Cross(forward, worldUp));

        Vec3 delta(0, 0, 0);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) delta = delta + forward * (moveSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) delta = delta - forward * (moveSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) delta = delta - right * (moveSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) delta = delta + right * (moveSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space))  delta = delta + worldUp * (moveSpeed * dt);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift)) delta = delta - worldUp * (moveSpeed * dt);

        airship.pos = airship.pos + delta;
        airship.pos.y = std::clamp(airship.pos.y, 3.0f, 40.0f);

        // Регулировка высоты камеры в режиме прицеливания
        if (aimingMode)
        {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::PageUp))
                aimCamHeight += 5.0f * dt;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::PageDown))
                aimCamHeight -= 5.0f * dt;
            aimCamHeight = std::clamp(aimCamHeight, 2.0f, 20.0f);
        }

        // ====== Управление прожектором ======
        bool spotlightToggleNow = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::L);
        if (spotlightToggleNow && !prevSpotlightToggle)
        {
            spotlightEnabled = !spotlightEnabled;
            std::cout << "Spotlight: " << (spotlightEnabled ? "ON" : "OFF") << std::endl;
        }
        prevSpotlightToggle = spotlightToggleNow;

        // ====== Сброс посылки ======
        bool dropNow = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Enter);
        if (dropNow && !prevDrop)
        {
            Package p;
            p.pos = airship.pos + Vec3(0.0f, -1.0f, 0.0f);
            p.vel = Vec3(0.0f, -25.0f, 0.0f);
            p.radius = 0.6f;
            p.alive = true;
            packages.push_back(p);
        }
        prevDrop = dropNow;

        // ====== Перегенерация сцены ======
        bool rNow = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::R);
        if (rNow && !prevR)
        {
            regenerateScene();
            packages.clear();
            hits = 0;
        }
        prevR = rNow;

        // ====== Обновление посылок и проверка коллизий ======
        for (auto& p : packages)
        {
            if (!p.alive) continue;

            p.pos = p.pos + p.vel * dt;

            if (p.pos.y <= 0.0f)
            {
                p.alive = false;
                continue;
            }

            for (auto& t : targets)
            {
                if (!t.alive) continue;
                if (SphereHit(p.pos, p.radius, t.pos + Vec3(0, 1.5f, 0), t.collisionRadius))
                {
                    t.alive = false;
                    p.alive = false;
                    hits++;
                    std::cout << "Hit! total=" << hits << "\n";
                    break;
                }
            }
        }

        packages.erase(
            std::remove_if(packages.begin(), packages.end(), [](const Package& p) { return !p.alive; }),
            packages.end()
        );

        // ======================= КАМЕРА =======================
        Vec3 camPos, camTarget;

        if (aimingMode)
        {
            camPos = airship.pos - worldUp * aimCamHeight;
            camTarget = Vec3(airship.pos.x, 0.0f, airship.pos.z) + forward * 2.0f;
        }
        else
        {
            camYaw = airship.rot.y;
            camPos = airship.pos - forward * camDist + worldUp * camHeight;
            camTarget = airship.pos + forward * 2.0f;
        }

        Mat4 view = Mat4::LookAt(camPos, camTarget, worldUp);

        // ======================= РЕНДЕР =======================
        glClearColor(0.03f, 0.05f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);

        // Глобальный направленный свет
        Vec3 lightDir = Normalize(Vec3(-1.0f, -1.2f, -0.2f));
        glUniform3f(uLightDirLoc, lightDir.x, lightDir.y, lightDir.z);
        glUniform3f(uLightColorLoc, 1.0f, 1.0f, 1.0f);
        glUniform3f(uAmbientLoc, 0.25f, 0.25f, 0.28f);

        // Параметры прожектора
        glUniform1i(uSpotlightEnabledLoc, spotlightEnabled ? 1 : 0);

        Vec3 spotlightPos = airship.pos - worldUp * 0.5f + forward * 1.0f;
        glUniform3f(uSpotlightPosLoc, spotlightPos.x, spotlightPos.y, spotlightPos.z);

        Vec3 spotlightDir = forward * 0.8f - worldUp * 0.2f;
        spotlightDir = Normalize(spotlightDir);
        glUniform3f(uSpotlightDirLoc, spotlightDir.x, spotlightDir.y, spotlightDir.z);

        glUniform3f(uSpotlightColorLoc, spotlightColor.x, spotlightColor.y, spotlightColor.z);
        glUniform1f(uSpotlightCutoffLoc, spotlightCutoff);
        glUniform1f(uSpotlightOuterCutoffLoc, spotlightOuterCutoff);
        glUniform1f(uSpotlightConstantLoc, spotlightConstant);
        glUniform1f(uSpotlightLinearLoc, spotlightLinear);
        glUniform1f(uSpotlightQuadraticLoc, spotlightQuadratic);

        glUniformMatrix4fv(uViewLoc, 1, GL_FALSE, view.m);
        glUniformMatrix4fv(uProjLoc, 1, GL_FALSE, proj.m);

        // Отрисовка объектов
        drawObj(ground);
        drawObj(tree);
        for (const auto& t : targets) drawObj(t);
        for (const auto& d : decos) drawObj(d);
        for (const auto& balloon : balloons) drawObj(balloon);
        drawObj(airship);

        // Посылки
        RenderObject pkgObj;
        pkgObj.mesh = &deco1Mesh;
        pkgObj.texDiffuse = deco1Tex;
        pkgObj.useNormal = false;
        pkgObj.useLight = false;
        pkgObj.isCloud = false;
        pkgObj.scale = { 0.5f, 0.5f, 0.5f };
        pkgObj.rot = { 0, 0, 0 };

        for (const auto& p : packages)
        {
            pkgObj.pos = p.pos;
            drawObj(pkgObj);
        }

        // Облака (рисуются последними из-за прозрачности)
        for (const auto& cloud : clouds) drawObj(cloud);

        glUseProgram(0);
        window.display();

        // Обновление заголовка окна
        static float titleTimer = 0.0f;
        titleTimer += dt;
        if (titleTimer > 0.3f)
        {
            titleTimer = 0.0f;
            std::string title = "Airship Delivery | hits=" + std::to_string(hits) +
                " | targets left=" + std::to_string(
                    std::count_if(targets.begin(), targets.end(),
                        [](const RenderObject& t) { return t.alive; }));

            if (aimingMode) title += " | AIMING MODE";
            if (spotlightEnabled) title += " | SPOTLIGHT ON";

            title += " | Clouds: " + std::to_string(clouds.size());
            title += " | Balloons: " + std::to_string(balloons.size());

            window.setTitle(title);
        }
    }

    // ======================= ОЧИСТКА РЕСУРСОВ =======================
    auto destroyMesh = [](Mesh& m)
        {
            if (m.VBO) glDeleteBuffers(1, &m.VBO);
            if (m.VAO) glDeleteVertexArrays(1, &m.VAO);
            m.VBO = 0; m.VAO = 0; m.vertexCount = 0;
        };

    destroyMesh(groundMesh);
    destroyMesh(airshipMesh);
    destroyMesh(treeMesh);
    destroyMesh(houseMesh);
    destroyMesh(deco1Mesh);
    destroyMesh(deco2Mesh);

    auto delTex = [](GLuint& t) { if (t) glDeleteTextures(1, &t); t = 0; };
    delTex(groundTex);
    delTex(airshipDiffuse);
    delTex(airshipNormal);
    delTex(treeDiffuse);
    delTex(houseDiffuse);
    delTex(houseLight);
    delTex(deco1Tex);
    delTex(deco2Tex);

    if (cloudTex != deco1Tex) delTex(cloudTex);
    if (balloonTex != deco2Tex) delTex(balloonTex);

    glDeleteProgram(prog);
    return 0;
}