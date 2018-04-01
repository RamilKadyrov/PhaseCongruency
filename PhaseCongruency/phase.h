#pragma once
//#include "options.h"
namespace cv
{
	class Mat;
	class _InputArray;
	class _OutputArray;
//  typedef const _InputArray& InputArray;
//	typedef const _OutputArray& OutputArray;
}

class PhaseCongruency
{

public:
	PhaseCongruency(cv::Size _img_size);
	~PhaseCongruency() {}
	void calc(cv::InputArray _src, cv::OutputArray _edges, cv::OutputArray _corners);
private:

    static const unsigned norient = 8;
    static const unsigned nscale = 4;

    const double sigma = -1.0 / (2.0 * log(0.65) * log(0.65));

    const float mult = 2.0f;
    const float minwavelength = 1.5f;
    const double epsilon = 0.0002; //0.0001
    const double cutOff = 0.4; //0.4;
    const double g = 10.0; 
    const double k = 10.0;

    cv::Mat	filter[nscale][norient];
};