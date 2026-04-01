#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <GL/gl.h>

#include "gl_texture.h"

#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

GlTexture2D::~GlTexture2D() {
    destroy();
}

bool GlTexture2D::update(int width, int height, const unsigned char* rgbaPixels) {
    if (width <= 0 || height <= 0 || !rgbaPixels)
        return false;

    if (!m_texId)
        glGenTextures(1, &m_texId);

    glBindTexture(GL_TEXTURE_2D, m_texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (m_width != width || m_height != height) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, rgbaPixels);
        m_width = width;
        m_height = height;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, rgbaPixels);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

void GlTexture2D::destroy() {
    if (m_texId) {
        glDeleteTextures(1, &m_texId);
        m_texId = 0;
    }
    m_width = 0;
    m_height = 0;
}
