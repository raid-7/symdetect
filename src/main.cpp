#include "symdetect.hpp"

#include <stdlib.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <opencv2/highgui.hpp>

const char* const WIN = "Display";

std::ostream& operator<<(std::ostream& stream, const symdetect::Contour& contour) {
    for (int i = 0; i < contour.size(); ++i) {
        const auto& point = contour[i];
        if (i != 0)
            stream << ' ';
        stream << '(' << point.x << "; " << point.y << ')';
    }
    return stream;
}


class DetectionCtx {
public:
    explicit DetectionCtx(const std::string& path) {
        image = cv::imread(path, 1);
        if (!image.data) {
            throw std::runtime_error("Cannot read image");
        }
    }

    void buildUi() {
        cv::createTrackbar("T1 [0..100]: ", WIN, &t1, 1024, tCallback, this);
        cv::createTrackbar("T2/T1 [2..3]: ", WIN, &t_ratio, 1024, tCallback, this);
        cv::createTrackbar("Poly [0..0.2]: ", WIN, &poly_acc, 1024, tCallback, this);
        cv::createTrackbar("Circle [0.5..1.0]: ", WIN, &circle_acc, 1024, tCallback, this);
    }

    void toggleGrayscale() {
        grayscale_only = !grayscale_only;
        compute();
    }

    void compute() {
        double t1d = t1 / 1024. * 100.;
        double ratio = 2. + t_ratio / 1024.;
        double t2d = t1d * ratio;
        double poly_acc_d = poly_acc / 1024. * 0.2;
        double circle_acc_d = 0.5 + circle_acc / 1024. * 0.5;

        symdetect::SymbolDetector detector(t1d, t2d, poly_acc_d, circle_acc_d, grayscale_only);

        cv::Mat outlined;
        auto result = detector.detect(image, outlined, debug_mode);
        cv::imshow(WIN, outlined);

        printResult(result);
    }

    void setDebugEnabled(bool enabled) {
        debug_mode = enabled;
        compute();
    }

private:
    static void printResult(const std::vector<symdetect::SquareWithCircles>& result) {
        for (const auto& sq : result) {
            std::cout << sq.square << '\n';
        }
        std::cout << std::endl;
    }

    static void tCallback(int, void* p) {
        auto& self = *static_cast<DetectionCtx*>(p);
        self.compute();
    }

private:
    cv::Mat image{};
    int t_ratio = 780;
    int t1 = 700;
    int poly_acc = 128;
    int circle_acc = 757;
    bool grayscale_only = false;
    bool debug_mode = false;
};

void usage() {
    std::cout << "Usage: symdetect [-e] [-v] <filename>" << std::endl;
    std::cout << "  -c    show parameter controls" << std::endl;
    std::cout << "  -i    paint intermediate stages" << std::endl;
    exit(1);
}

int main(int argc, char* argv[]) {
    const char* fname;
    bool enable_controls = false;
    bool enable_intermediate = false;

    for (int c; (c = getopt(argc, argv, "ci")) != -1;) {
        switch (c) {
            case 'c':
                enable_controls = true;
                break;
            case 'i':
                enable_intermediate = true;
                break;
            case '?':
                usage();
                break;
        }
    }

    if (optind + 1 != argc) {
        usage();
    }
    fname = argv[optind];

    DetectionCtx ctx{fname};
    cv::namedWindow(WIN, cv::WINDOW_NORMAL);
    cv::resizeWindow(WIN, 1024, 768);
    if (enable_controls)
        ctx.buildUi();
    ctx.setDebugEnabled(enable_intermediate);

    int key_code;
    do {
        key_code = cv::waitKey();
        switch (key_code) {
            case 71:
                ctx.toggleGrayscale();
                break;
            case 72:
                enable_intermediate = !enable_intermediate;
                ctx.setDebugEnabled(enable_intermediate);
                break;
        }
    } while (key_code != 27 && key_code != -1);
    return 0;
}
