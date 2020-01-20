#version 400

layout (location = 0) out vec4 FragColor;

in vec2 texCoord;
uniform sampler2D textureSampler;

void main() {
    vec2 uv = (1.0 + texCoord) / 2.0;
    // FragColor = vec4(uv, 0.0, 1.0);
    FragColor = texture(textureSampler, uv);
}