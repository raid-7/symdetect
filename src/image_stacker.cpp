#include "symdetect.hpp"

#include <opencv2/imgproc.hpp>

namespace symdetect {

ImageStacker& ImageStacker::operator<<(const Mat& oth) {
    images.emplace_back();
    if (oth.channels() == 1) {
        Mat tmp;
        cv::cvtColor(oth, tmp, cv::COLOR_GRAY2BGR);
        tmp.convertTo(images.back(), CV_8U);
    } else {
        oth.convertTo(images.back(), CV_8U);
    }
    return *this;
}

Mat ImageStacker::get() const {
    Mat res;
    cv::hconcat(images.data(), images.size(), res);
    return res;
}

Mat& ImageStacker::with(const Mat& oth) {
    *this << oth;
    return images.back();
}

}
