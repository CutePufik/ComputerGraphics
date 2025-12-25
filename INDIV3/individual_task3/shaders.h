#ifndef SHADERS_H
#define SHADERS_H

const char* vertexShaderSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTex;
layout(location=3) in vec3 aTangent;

uniform mat4 uModel; // из локальных координат модели в мировые
uniform mat4 uView;  // из мира в координаты камеры
uniform mat4 uProj;  // проекция (перспектива)

out VS_OUT {
    vec2 uv;
    vec3 worldPos;
    vec3 worldNormal;
    mat3 TBN;  // для normal map
} vs_out;

void main()
{
    vec3 T = normalize(mat3(uModel) * aTangent);
    vec3 N = normalize(mat3(uModel) * aNormal);
    T = normalize(T - dot(T,N)*N);
    vec3 B = cross(N, T);

    vs_out.TBN = mat3(T, B, N);
    vec4 wp = uModel * vec4(aPos,1.0);
    vs_out.worldPos = wp.xyz;
    vs_out.worldNormal = N;
    vs_out.uv = aTex;

    gl_Position = uProj * uView * wp;
}
)";

const char* fragmentShaderSrc = R"(
#version 330 core

in VS_OUT {
    vec2 uv;
    vec3 worldPos;
    vec3 worldNormal;
    mat3 TBN;
} fs_in;

out vec4 FragColor;

uniform sampler2D uDiffuse;  // основная (цветовая) текстура
uniform sampler2D uNormalMap;  // карта нормалей
uniform sampler2D uLightMap;  // карта освещения

uniform bool uUseNormalMap;
uniform bool uUseLightMap;
uniform bool uIsCloud;

uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;

// Прожектор
uniform bool uSpotlightEnabled;
uniform vec3 uSpotlightPos;
uniform vec3 uSpotlightDir;
uniform vec3 uSpotlightColor;
uniform float uSpotlightCutoff;
uniform float uSpotlightOuterCutoff;
uniform float uSpotlightConstant;
uniform float uSpotlightLinear;
uniform float uSpotlightQuadratic;

void main()
{
    vec4 texColor = texture(uDiffuse, fs_in.uv);
    
    // Если это облако - используем прозрачность
    if(uIsCloud)
    {
        // Для облаков используем альфа-канал текстуры
        // Если текстура не имеет альфа-канала, используем фиксированную прозрачность
        float alpha = texColor.a;
        if(alpha < 0.01) alpha = 0.7;
        
        
        // Облака почти не реагируют на направленный свет
        vec3 lighting = uAmbientColor * 1.5;
        
        // Мягкие облака - используем только ambient освещение
        vec3 color = texColor.rgb * lighting;
        FragColor = vec4(color, alpha * 0.8);
        return;
    }
    
    // Обычное освещение для всех остальных объектов
    vec3 albedo = texColor.rgb;

    vec3 N = normalize(fs_in.worldNormal);
    if(uUseNormalMap)
    {
        vec3 nrm = texture(uNormalMap, fs_in.uv).xyz * 2.0 - 1.0;
        N = normalize(fs_in.TBN * nrm);
    }

    vec3 L = normalize(-uLightDir);
    float ndotl = max(dot(N, L), 0.0);

    vec3 lighting = uAmbientColor + uLightColor * ndotl;

    // Добавляем прожектор
    if(uSpotlightEnabled)
    {
        vec3 lightDir = normalize(uSpotlightPos - fs_in.worldPos);
        float theta = dot(lightDir, normalize(-uSpotlightDir));
        float epsilon = uSpotlightCutoff - uSpotlightOuterCutoff;
        float intensity = clamp((theta - uSpotlightOuterCutoff) / epsilon, 0.0, 1.0);
        
        if(theta > uSpotlightOuterCutoff)  // попадает ли точка вообще в область прожектора
        {
            float distance = length(uSpotlightPos - fs_in.worldPos);
            //затухание по расстоянию
            float attenuation = 1.0 / (uSpotlightConstant + uSpotlightLinear * distance + uSpotlightQuadratic * (distance * distance));
            
            float spotlightDiffuse = max(dot(N, lightDir), 0.0);
            vec3 spotlightContribution = uSpotlightColor * spotlightDiffuse * attenuation * intensity;
            lighting += spotlightContribution;
        }
    }

    if(uUseLightMap)
    {
        vec3 lm = texture(uLightMap, fs_in.uv).rgb;
        lighting *= lm;
    }

    vec3 color = albedo * lighting;
    FragColor = vec4(color, 1.0);
}
)";

#endif 