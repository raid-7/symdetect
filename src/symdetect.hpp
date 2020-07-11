#pragma once

#include <vector>
#include <opencv2/imgproc.hpp>

namespace symdetect {

using cv::Mat;
using cv::Point;
using Contour = std::vector<Point>;

namespace geom {
class Segment {
public:
    Segment() = default;

    Segment(Point a, Point b);

    int lengthSq() const;

    float length() const;

    static double angleDeg(const Segment& u, const Segment& v);

private:
    cv::Point a{};
    cv::Point b{};
};

class Circle {
public:
    Circle() = default;

    explicit Circle(cv::Vec3f c);

    Circle(float x, float y, float radius);

    Point iCenter() const;

    int iRadius() const;

    cv::Point2f center() const;

    Circle move(const cv::Point2f& offset) const;

private:
    float x, y, radius;
};
}

class ImageStacker {
public:
    ImageStacker& operator<<(const Mat& oth);

    Mat get() const;

    Mat& with(const Mat& oth);

private:
    std::vector<Mat> images;
};


struct SquareWithCircles {
    Contour square;
    std::vector<geom::Circle> circles;

    float sideLength() const;

    cv::Point2f center() const;
};

class SymbolDetector {
public:
    explicit SymbolDetector(
            double t1 = 1., double t2 = 5.,
            double poly_acc = 0.02,
            double circle_acc = 0.87,
            bool grayscale_only = false
    );

    std::vector<SquareWithCircles> detect(const Mat& source, cv::InputOutputArray dst, bool debug_mode = false) const;

private:
    static void drawResult(const cv::_InputOutputArray& dst, const std::vector<SquareWithCircles>& data,
                           bool paint_circles, int thickness);

    static Mat filter(const Mat& source, const cv::Size& target_size);

    Mat canny(const Mat& source) const;

    static bool isQuadSquare(const Contour& quad) ;

    static Mat getContourRegionSlice(const Contour& contour, const Mat& source);

    static bool isQuad(const Contour& contour, const cv::Size& im_size);

    static std::vector<Contour> findContours(const Mat& source);

    static bool isInside(const Contour& inner, const Contour& outer);

    static std::vector<Contour> removeInnerQuads(const std::vector<Contour>& quads);

    std::vector<Contour> findSquares(const std::vector<Contour>& contours, const cv::Size& im_size) const;

    static void drawCircles(cv::InputOutputArray img, const std::vector<geom::Circle>& circles,
                            const cv::Point& offset = {0, 0}, int thickness = 1);

    std::vector<geom::Circle> findCircles(const Mat& source) const;

    std::vector<SquareWithCircles> squaresWithCircles(const Mat& source, const std::vector<Contour>& squares) const;

    static float sortingParameter(const SquareWithCircles& sq);

    static void sortSquares(std::vector<SquareWithCircles>& squares);

private:
    double t1 = 1., t2 = 5.;
    double poly_acc = 0.02;
    double circle_acc = 0.87;
    bool grayscale_only = false;

    static constexpr double square_side_ratio_max = 1.15;
    static constexpr double square_angle_min = 80; // deg
    static constexpr double square_angle_max = 100; // deg
};
}
