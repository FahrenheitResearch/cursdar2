#pragma once

class App;

namespace ui {
    void init();
    void render(App& app);
    bool wantsMouseCapture();
    void shutdown();
}
