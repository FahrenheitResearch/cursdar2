#pragma once

class GlTexture2D {
public:
    GlTexture2D() = default;
    ~GlTexture2D();

    bool update(int width, int height, const unsigned char* rgbaPixels);
    void destroy();

    unsigned int textureId() const { return m_texId; }
    int width() const { return m_width; }
    int height() const { return m_height; }

private:
    unsigned int m_texId = 0;
    int m_width = 0;
    int m_height = 0;
};
