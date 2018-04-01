#pragma once

namespace cv
{
	class Mat;
	class _InputArray;
	class _OutputArray;
    typedef const _InputArray& InputArray;
    typedef const _OutputArray& OutputArray;
}

class PhaseCongruency
{
public:
	PhaseCongruency(cv::Size _img_size, size_t _nscale, size_t _norient);
	~PhaseCongruency() {}
	void calc(cv::InputArray _src);
    void feature(cv::InputArray _src, cv::OutputArray _edges, cv::OutputArray _corners);

private:
    cv::Size size;
    size_t norient;
    size_t nscale;

    const double sigma = -1.0 / (2.0 * log(0.65) * log(0.65));

    const float mult = 2.0f;
    const float minwavelength = 1.5f;
    const double epsilon = 0.0002; //0.0001
    const double cutOff = 0.4; //0.4;
    const double g = 10.0; 
    const double k = 10.0;

    std::vector<cv::Mat>	filter;
    std::vector<cv::Mat>        pc;
};