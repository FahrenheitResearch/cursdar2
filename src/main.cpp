#ifdef _WIN32
#include <windows.h>
#endif

#include "app.h"
#include "ui/ui.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdio>
#include <chrono>
#include <thread>

// ── Globals for callbacks ───────────────────────────────────
static App* g_app = nullptr;
static bool g_mouseDown = false;
static bool g_rightMouseDown = false;
static bool g_middleMouseDown = false;
static double g_lastMouseX = 0, g_lastMouseY = 0;
static double g_leftMouseDownX = 0, g_leftMouseDownY = 0;

static void scrollCallback(GLFWwindow* window, double xoff, double yoff) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (g_app) g_app->onScroll(xoff, yoff);
}

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (g_app && width > 0 && height > 0) {
        g_app->onFramebufferResize(width, height);
    }
}

static void windowSizeCallback(GLFWwindow* window, int width, int height) {
    if (g_app && width > 0 && height > 0) {
        g_app->onResize(width, height);
    }
}

int main(int argc, char** argv) {
    printf("=== CURSDAR2 - CUDA Radar Workstation ===\n");
    printf("Loading next-generation shell on the CUDA radar engine\n\n");
#ifdef _WIN32
    SetProcessDPIAware();
#endif

    // ── GLFW init ───────────────────────────────────────────
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    int winW = mode->width;
    int winH = mode->height;

    GLFWwindow* window = glfwCreateWindow(winW, winH, "CURSDAR2", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // No vsync - maximum frame rate
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetWindowSizeCallback(window, windowSizeCallback);

    auto syncWindowMetrics = [&](int& logicalW, int& logicalH, int& framebufferW, int& framebufferH) {
        glfwGetWindowSize(window, &logicalW, &logicalH);
        glfwGetFramebufferSize(window, &framebufferW, &framebufferH);
        if (logicalW <= 0 || logicalH <= 0) {
            logicalW = (framebufferW > 0) ? framebufferW : 1;
            logicalH = (framebufferH > 0) ? framebufferH : 1;
        }
        if (framebufferW <= 0 || framebufferH <= 0) {
            framebufferW = logicalW;
            framebufferH = logicalH;
        }
    };

    auto settleWindowMetrics = [&](int& logicalW, int& logicalH, int& framebufferW, int& framebufferH) {
        int lastFramebufferW = -1;
        int lastFramebufferH = -1;
        for (int attempt = 0; attempt < 6; attempt++) {
            glfwPollEvents();
            syncWindowMetrics(logicalW, logicalH, framebufferW, framebufferH);
            if (framebufferW > 0 && framebufferH > 0 &&
                framebufferW == lastFramebufferW &&
                framebufferH == lastFramebufferH) {
                break;
            }
            lastFramebufferW = framebufferW;
            lastFramebufferH = framebufferH;
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    };

    int fbW = winW;
    int fbH = winH;
    settleWindowMetrics(winW, winH, fbW, fbH);

    // ── Dear ImGui init ─────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    ui::init();

    int exitCode = 0;

    // ── App init ────────────────────────────────────────────
    {
    App app;
    g_app = &app;

    if (!app.init(winW, winH, fbW, fbH)) {
        fprintf(stderr, "Failed to initialize app\n");
        g_app = nullptr;
        exitCode = 1;
    } else {
        settleWindowMetrics(winW, winH, fbW, fbH);
        app.onResize(winW, winH);
        app.onFramebufferResize(fbW, fbH);
        printf("Starting main loop...\n");

    // ── Main loop ───────────────────────────────────────────
    auto lastFrame = std::chrono::steady_clock::now();
    int frameCount = 0;
    float fpsTimer = 0;
    float fps = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        int currentWinW = 0;
        int currentWinH = 0;
        int currentFbW = 0;
        int currentFbH = 0;
        syncWindowMetrics(currentWinW, currentWinH, currentFbW, currentFbH);
        if (currentWinW != app.viewport().width || currentWinH != app.viewport().height)
            app.onResize(currentWinW, currentWinH);
        if (currentFbW != app.framebufferWidth() || currentFbH != app.framebufferHeight())
            app.onFramebufferResize(currentFbW, currentFbH);

        // Calculate delta time
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - lastFrame).count();
        lastFrame = now;

        // FPS counter
        frameCount++;
        fpsTimer += dt;
        if (fpsTimer >= 1.0f) {
            fps = frameCount / fpsTimer;
            frameCount = 0;
            fpsTimer = 0;
            std::string activeStation = app.activeStationName();
            char title[144];
            snprintf(title, sizeof(title),
                     "CURSDAR2 - %s | %s | Tilt %.1f | %d stations | %.0f FPS",
                     activeStation.c_str(),
                     PRODUCT_INFO[app.activeProduct()].name,
                     app.activeTiltAngle(),
                     app.stationsLoaded(), fps);
            glfwSetWindowTitle(window, title);
        }

        // Mouse tracking
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);

        const bool uiCapturingMouse = ImGui::GetIO().WantCaptureMouse || ui::wantsMouseCapture();

        // Station tracking (lock station in cross-section/3D mode)
        if (!uiCapturingMouse && !app.crossSection() && !app.mode3D()) {
            app.onMouseMove(mx, my);
        }

        // Left drag to pan, right drag to orbit (3D mode)
        if (!uiCapturingMouse) {
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                if (g_mouseDown)
                    app.onMouseDrag(mx - g_lastMouseX, my - g_lastMouseY);
                else {
                    g_leftMouseDownX = mx;
                    g_leftMouseDownY = my;
                }
                g_mouseDown = true;
            } else {
                if (g_mouseDown && !app.crossSection() && !app.mode3D()) {
                    const double clickDx = mx - g_leftMouseDownX;
                    const double clickDy = my - g_leftMouseDownY;
                    const double clickDistSq = clickDx * clickDx + clickDy * clickDy;
                    if (clickDistSq <= 36.0) {
                        const int hitIdx = app.stationAtScreen(mx, my);
                        if (hitIdx >= 0)
                            app.selectStation(hitIdx, false);
                    }
                }
                g_mouseDown = false;
            }
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
                if (g_rightMouseDown)
                    app.onRightDrag(mx - g_lastMouseX, my - g_lastMouseY);
                else if (app.crossSection())
                    app.onMiddleClick(mx, my); // start cross-section line
                g_rightMouseDown = true;
            } else {
                g_rightMouseDown = false;
            }
            // Right-drag also moves cross-section endpoint
            if (g_rightMouseDown && app.crossSection()) {
                app.onMiddleDrag(mx, my);
            }
        }
        g_lastMouseX = mx;
        g_lastMouseY = my;

        // Update app state
        app.update(dt);

        // Render radar to GPU texture
        app.render();

        // ── ImGui frame ─────────────────────────────────────
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ui::render(app);

        // FPS overlay
        ImGui::SetNextWindowPos(ImVec2((float)app.viewport().width - 100, (float)app.viewport().height - 30));
        ImGui::Begin("##fps", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoBackground);
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 0.8f), "%.0f FPS", fps);
        ImGui::End();

        ImGui::Render();

        glViewport(0, 0, app.framebufferWidth(), app.framebufferHeight());
        glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // ── Cleanup ─────────────────────────────────────────────
    g_app = nullptr;
    // App destructor handles GPU cleanup
    }

    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    printf("CURSDAR2 shutdown complete.\n");
    return exitCode;
}
